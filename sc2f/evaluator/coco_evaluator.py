import json
import tempfile
from xml.parsers.expat import model
import torch
from data.coco import *
from data.coco_fb import *
from data.coco_fb_diff import *

import wandb

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


from utils.CocoEval_with_Difficulty import COCOeval as COCOeval_with_difficulty
from utils.center_error import eval_center_error, eval_center_error_lean

import json, tempfile, os, ctypes, gc
import numpy as np

def _trim_os_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, testset=False, transform=None, EAD=False, image_folder=None, ann_file=None, iouThr=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
        if self.testset:
            image_set = 'test2017'
        else:
            image_set = 'val2017'
        if ann_file is not None:
            if EAD:
                self.dataset = COCODataset_FB_Diff(
                            data_dir=data_dir,
                            image_folder=image_folder,
                            transform=None,
                            ann_file=ann_file,)
            else:
                self.dataset = COCODataset_FB(
                                data_dir=data_dir,
                                image_folder=image_folder,
                                transform=None,
                                ann_file=ann_file,)
        else:
            self.dataset = COCODataset(
                                data_dir=data_dir,
                                image_set=image_set,
                                transform=None)
        self.transform = transform
        self.device = device
        self.EAD = EAD

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.
        self.iouThr = iouThr

    def evaluate(self, model, max_images: int | None = None):
        """
        COCO evaluation with lower RAM usage.
        - Streams detections to a temporary JSON file (no giant Python list).
        - Optional 'max_images' to run on a subset for sanity checks.
        """
        model.eval()
        device = self.device
        num_images_total = len(self.dataset)
        num_images = num_images_total if max_images is None else min(max_images, num_images_total)
        print(f'total number of images: {num_images_total} (evaluating {num_images})')

        ids = []
        # open a streaming JSON file to avoid a massive in-memory list
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(tmp_fd)  # we'll open it with normal Python IO
        wrote_any = False
        with open(tmp_path, 'w') as fjson, torch.inference_mode():
            fjson.write('[')

            for index in range(num_images):
                if index % 500 == 0:
                    print(f'[Eval: {index} / {num_images}]')

                # load an image (H,W,3), keep as numpy only long enough for transform
                img, id_ = self.dataset.pull_image(index)
                h, w, _ = img.shape
                orig_size = np.array([[w, h, w, h]], dtype=np.float32)

                # preprocess -> (1,C,H,W) on device
                x = self.transform(img)[0].unsqueeze(0).to(device, non_blocking=False)

                id_ = int(id_)
                ids.append(id_)

                # inference
                outputs = model(x)
                if isinstance(outputs, dict):
                    bboxes = outputs['bboxes']   # (N,4) on device
                    scores = outputs['scores']   # (N,)
                    cls_inds = outputs['labels'] # (N,)
                    token_idx = outputs.get('token_idx', None)
                else:
                    bboxes, scores, cls_inds = outputs
                    token_idx = None

                # Build scale on the same device/dtype as bboxes to avoid mismatch
                if isinstance(bboxes, np.ndarray):
                    # rare path: convert to torch first
                    bboxes = torch.from_numpy(bboxes).to(self.device)

                # Ensure 2D shape (N,4). If your model occasionally returns (4,) for 1 det:
                if bboxes.ndim == 1:
                    bboxes = bboxes.unsqueeze(0)

                scale = bboxes.new_tensor([w, h, w, h]).unsqueeze(0)  # (1,4) on same device/dtype
                if self.EAD:
                    scale = scale * 3.0

                bboxes = bboxes * scale  # (N,4) * (1,4) -> (N,4)

                # move only what we need to CPU as numpy; keep per-detection loop short
                b = bboxes.detach().cpu().numpy()
                s = scores.detach().cpu().numpy() if not isinstance(scores, np.ndarray) else scores
                c = cls_inds.detach().cpu().numpy() if not isinstance(cls_inds, np.ndarray) else cls_inds
                t = None if token_idx is None else (token_idx.detach().cpu().numpy() if not isinstance(token_idx, np.ndarray) else token_idx)

                # append detections directly into streaming JSON
                for i in range(b.shape[0]):
                    x1, y1, x2, y2 = float(b[i,0]), float(b[i,1]), float(b[i,2]), float(b[i,3])
                    label = self.dataset.class_ids[int(c[i])]
                    det = {
                        "image_id": id_,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s[i])
                    }
                    if t is not None:
                        det["token_idx"] = int(t[i])

                    # write with comma separators to keep a valid JSON array
                    if wrote_any: fjson.write(',')
                    json.dump(det, fjson, separators=(',', ':'))
                    wrote_any = True

                # free per-iter tensors
                del x, outputs, bboxes, scores, cls_inds, token_idx
                _trim_os_memory()

            fjson.write(']')  # close JSON array

        print('evaluating ......')
        cocoGt = self.dataset.coco
        if self.testset:
            cocoDt = cocoGt.loadRes(tmp_path)
            return -1, -1
        else:
            cocoDt = cocoGt.loadRes(tmp_path)
            if self.iouThr is not None:
                cocoEval = COCOeval_with_difficulty(cocoGt, cocoDt, 'bbox', iouThr=self.iouThr)
            else:
                cocoEval = COCOeval_with_difficulty(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50, ap75 = cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[2]
            ap_normal, ap_easy, ap_medium, ap_hard = cocoEval.stats[12], cocoEval.stats[13], cocoEval.stats[14], cocoEval.stats[15]
            print(f'ap50_95: {ap50_95}, ap50: {ap50}, ap75: {ap75}')
            print(f'ap_normal: {ap_normal}, ap_easy: {ap_easy}, ap_medium: {ap_medium}, ap_hard: {ap_hard}')

            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50

            # --- center error (memory-lean version below) ---
            center_error = eval_center_error_lean(['bipartite_mean_se_img_wise'], tmp_path, self.dataset, EAD=self.EAD)

            for key, value in center_error.items():
                print(f'center error eval type: {key}')
                for k, v in value.items():
                    print(f'{k} : {v}')

            result_dict = {
                'ap50_95': ap50_95,
                'ap50': ap50,
                'ap75': ap75,
                'ap_normal': ap_normal,
                'ap_easy': ap_easy,
                'ap_medium': ap_medium,
                'ap_hard': ap_hard,
                'ap_outside': cocoEval.stats[16] if len(cocoEval.stats) > 16 else -1,
                'center_errors': center_error['bipartite_mean_se_img_wise']['center_errors'],
                'center_errors_normal': center_error['bipartite_mean_se_img_wise']['center_errors_normal'],
                'center_errors_easy': center_error['bipartite_mean_se_img_wise']['center_errors_easy'],
                'center_errors_medium': center_error['bipartite_mean_se_img_wise']['center_errors_medium'],
                'center_errors_hard': center_error['bipartite_mean_se_img_wise']['center_errors_hard'],
                'center_errors_outside': center_error['bipartite_mean_se_img_wise']['center_errors_outside'],
            }

            # cleanup temp file
            try: os.remove(tmp_path)
            except: pass

            return result_dict

