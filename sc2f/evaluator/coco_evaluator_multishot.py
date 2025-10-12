import json
import tempfile
from xml.parsers.expat import model
import torch
from data.coco import *
from data.coco_fb import *
from data.coco_fb_diff import *

import wandb
import glob, random, cv2

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
        Multishot-aware COCO evaluation (streaming JSON, low RAM).
        Does NOT call dataset.pull_image(); instead it loads shots from
        image_folder/{id:012d}/* and unions detections across those shots.
        """
        model.eval()
        device = self.device

        # Resolve COCO ids in dataset order
        if hasattr(self.dataset, 'ids'):
            coco_ids_order = list(map(int, self.dataset.ids))
        elif hasattr(self.dataset, 'img_ids'):
            coco_ids_order = list(map(int, self.dataset.img_ids))
        else:
            coco_ids_order = list(map(int, self.dataset.coco.getImgIds()))

        num_images_total = len(coco_ids_order)
        num_images = num_images_total if max_images is None else min(max_images, num_images_total)
        print(f'total number of images: {num_images_total} (evaluating {num_images})')

        # evaluator knobs (can be set by caller)
        num_per_index  = int(getattr(self, 'num_per_index', 0) or 0)  # 0 => use all
        seed_per_eval  = int(getattr(self, 'seed', 123))
        cand_exts      = list(getattr(self, 'candidate_exts', ['.jpg', '.jpeg', '.png']))
        nms_per_class  = bool(getattr(self, 'nms_per_class', False))

        # model thresholds as configured by the runner
        conf_sel = float(getattr(model, 'conf_thresh', 0.0) or 0.0)
        nms_sel  = float(getattr(model, 'nms_thresh', 1.0) or 1.0)
        topk_sel = getattr(model, 'topk', None)
        if topk_sel is not None:
            topk_sel = int(topk_sel)

        # torchvision NMS (fallback to a tiny CPU NMS if unavailable)
        try:
            from torchvision.ops import nms as tv_nms
        except Exception:
            tv_nms = None

        def _nms_cpu(boxes_t, scores_t, thr):
            if boxes_t.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            if tv_nms is not None:
                return tv_nms(boxes_t, scores_t, float(thr)).cpu()
            keep = []
            idxs = scores_t.argsort(descending=True)
            while idxs.numel() > 0:
                i = idxs[0].item()
                keep.append(i)
                if idxs.numel() == 1:
                    break
                aa = boxes_t[i]
                bb = boxes_t[idxs[1:]]
                lt = torch.maximum(aa[:2], bb[:, :2])
                rb = torch.minimum(aa[2:],  bb[:, 2:])
                wh = (rb - lt).clamp(min=0)
                inter = wh[:, 0] * wh[:, 1]
                area_a = ((aa[2] - aa[0]).clamp(min=0) * (aa[3] - aa[1]).clamp(min=0))
                area_b = ((bb[:, 2] - bb[:, 0]).clamp(min=0) * (bb[:, 3] - bb[:, 1]).clamp(min=0))
                iou = inter / (area_a + area_b - inter + 1e-12)
                idxs = idxs[1:][iou <= float(thr)]
            return torch.tensor(keep, dtype=torch.long)

        ids_for_eval = []
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(tmp_fd)
        wrote_any = False

        with open(tmp_path, 'w') as fjson, torch.inference_mode():
            fjson.write('[')

            image_root = getattr(self.dataset, 'img_folder', None)
            if image_root is None:
                raise RuntimeError("dataset.img_folder must be set for multishot evaluation.")

            for i_idx in range(num_images):
                if i_idx % 500 == 0:
                    print(f'[Eval: {i_idx} / {num_images}]')

                img_id = coco_ids_order[i_idx]
                ids_for_eval.append(img_id)

                # ---- enumerate candidate shots: image_folder/{id:012d}/*.{ext} ----
                subdir = os.path.join(image_root, f"{img_id:012d}")
                cand_paths = []
                if os.path.isdir(subdir):
                    for e in cand_exts:
                        cand_paths.extend(glob.glob(os.path.join(subdir, f"*{e}")))
                cand_paths = sorted(cand_paths)

                # Fallback: if folder missing/empty, try a flat image like {id:012d}.{ext}
                if not cand_paths:
                    for e in cand_exts:
                        p = os.path.join(image_root, f"{img_id:012d}{e}")
                        if os.path.isfile(p):
                            cand_paths = [p]
                            break

                if not cand_paths:
                    # no candidates; emit nothing for this id
                    continue

                # optional subsample (deterministic)
                if num_per_index > 0 and len(cand_paths) > num_per_index:
                    rng = random.Random(seed_per_eval + img_id)
                    cand_paths = rng.sample(cand_paths, num_per_index)

                # ---- run model on each shot, collect union on CPU ----
                all_boxes, all_scores, all_labels, all_tokens = [], [], [], []
                for p in cand_paths:
                    im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                    if im_bgr is None:
                        continue
                    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                    h0, w0 = im_rgb.shape[:2]

                    # preprocess
                    x = self.transform(im_rgb)[0].unsqueeze(0).to(device, non_blocking=False)

                    # forward
                    outputs = model(x)
                    if isinstance(outputs, dict):
                        bboxes = outputs['bboxes']
                        scores = outputs['scores']
                        cls_inds = outputs['labels']
                        token_idx = outputs.get('token_idx', None)
                    else:
                        bboxes, scores, cls_inds = outputs
                        token_idx = None

                    # coerce to torch on CPU
                    if isinstance(bboxes, np.ndarray):
                        bboxes = torch.from_numpy(bboxes).to(device)
                    if bboxes.ndim == 1:
                        bboxes = bboxes.unsqueeze(0)

                    scale = torch.tensor([w0, h0, w0, h0], dtype=torch.float32)
                    if self.EAD:
                        scale = scale * 3.0
                    b_cpu = (bboxes.detach().to(torch.float32).cpu() * scale)  # (N,4) xyxy

                    s_cpu = scores.detach().to(torch.float32).cpu() if isinstance(scores, torch.Tensor) \
                            else torch.tensor(scores, dtype=torch.float32)
                    c_cpu = cls_inds.detach().to(torch.int64).cpu() if isinstance(cls_inds, torch.Tensor) \
                            else torch.tensor(cls_inds, dtype=torch.int64)

                    all_boxes.append(b_cpu)
                    all_scores.append(s_cpu)
                    all_labels.append(c_cpu)

                    if token_idx is not None:
                        t_cpu = token_idx.detach().to(torch.int64).cpu() if isinstance(token_idx, torch.Tensor) \
                                else torch.tensor(token_idx, dtype=torch.int64)
                        all_tokens.append(t_cpu)

                    del x, outputs, bboxes, scores, cls_inds, token_idx
                    _trim_os_memory()

                # concat union
                if len(all_boxes):
                    boxes  = torch.cat(all_boxes,  dim=0)
                    scores = torch.cat(all_scores, dim=0)
                    labels = torch.cat(all_labels, dim=0)
                    tokens = torch.cat(all_tokens, dim=0) if len(all_tokens) == len(all_boxes) else None
                else:
                    boxes  = torch.zeros((0,4), dtype=torch.float32)
                    scores = torch.zeros((0,),   dtype=torch.float32)
                    labels = torch.zeros((0,),   dtype=torch.int64)
                    tokens = None

                # post-hoc filtering: conf -> NMS -> top-k
                if boxes.numel() and scores.numel():
                    if conf_sel > 0:
                        m = scores >= conf_sel
                        boxes, scores, labels = boxes[m], scores[m], labels[m]
                        if tokens is not None: tokens = tokens[m]
                    if boxes.size(0):
                        if nms_per_class and labels.numel():
                            kept = []
                            for cls in labels.unique():
                                sel = (labels == cls)
                                if sel.sum() == 0:
                                    continue
                                idx = _nms_cpu(boxes[sel], scores[sel], nms_sel)
                                kept.append(torch.nonzero(sel, as_tuple=False).squeeze(1)[idx])
                            keep = torch.cat(kept, dim=0) if len(kept) else torch.empty(0, dtype=torch.long)
                        else:
                            keep = _nms_cpu(boxes, scores, nms_sel)
                        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                        if tokens is not None: tokens = tokens[keep]
                    if topk_sel is not None and boxes.size(0) > topk_sel:
                        order = torch.argsort(scores, descending=True)[:topk_sel]
                        boxes, scores, labels = boxes[order], scores[order], labels[order]
                        if tokens is not None: tokens = tokens[order]

                # stream to JSON (COCO expects XYWH)
                for i in range(boxes.size(0)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    det = {
                        "image_id": img_id,
                        "category_id": int(self.dataset.class_ids[int(labels[i].item())]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(scores[i].item())
                    }
                    if tokens is not None:
                        det["token_idx"] = int(tokens[i].item())
                    if wrote_any: fjson.write(',')
                    json.dump(det, fjson, separators=(',', ':'))
                    wrote_any = True

                del boxes, scores, labels, tokens, all_boxes, all_scores, all_labels, all_tokens
                _trim_os_memory()

            fjson.write(']')

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
            cocoEval.params.imgIds = ids_for_eval
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

            try:
                os.remove(tmp_path)
            except:
                pass

            return result_dict

