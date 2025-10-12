from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm

img_difficulty_to_num = {
    'normal': 0,
    'easy': 1,
    'medium': 2,
    'hard': 3
}



def center_dis_given_ann_pred(ann, pred, max_length):
    assert pred['category_id'] == ann['category_id'], "category_id mismatch"
    pred_bbox = pred['bbox']
    ann_bbox = ann['bbox']
    pred_bbox = [pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2]
    ann_bbox = [ann_bbox[0] + ann_bbox[2]/2, ann_bbox[1] + ann_bbox[3]/2]
    center_distance = np.linalg.norm(np.array(pred_bbox) - np.array(ann_bbox)) / max_length
    return center_distance

def center_dis_matrix_given_anns_preds(anns, preds, max_length):
    center_dis_matrix = np.zeros((len(anns), len(preds)))
    diff_labels = np.zeros((len(anns), len(preds)), dtype=int)

    for idx, ann in enumerate(anns):
        for jdx, pred in enumerate(preds):
            center_distance = center_dis_given_ann_pred(ann, pred, max_length)
            center_dis_matrix[idx, jdx] = center_distance
            diff_labels[idx, jdx] = img_difficulty_to_num[ann['difficulty']]

    return center_dis_matrix, diff_labels

def evaluate_center_dis_bipartite(imageid2pred, imageid2ann, imageid2size, isface=True):
    center_errors = []

    center_errors_normal = []
    center_errors_easy = []
    center_errors_medium = []
    center_errors_hard = []
    center_errors_outside = []

    tot_num = len(imageid2pred.keys())
    num_print = tot_num // 50

    for iter_id, (k, v) in enumerate(imageid2pred.items()):

        if iter_id % num_print == 0:
            print(f'Processing {iter_id}th/{tot_num} image...')

        # A k means a image_id and v means a list of predictions
        imageid2pred[k] = sorted(v, key=lambda x: x['score'], reverse=True)

        width, height = imageid2size[k]
        max_length = np.sqrt(width**2 + height**2)

        center_errors_per_img = []

        try: 
            category_id = 0 if isface else 1
            face_anns = [ann for ann in imageid2ann[k] if ann['category_id'] == category_id]
            face_preds = [pred for pred in imageid2pred[k] if pred['category_id'] == category_id]

            center_distance_matrix, diff_label = center_dis_matrix_given_anns_preds(face_anns, face_preds, max_length)

            row_ind, col_ind = linear_sum_assignment(center_distance_matrix)

            center_distance_min_per_img = center_distance_matrix[row_ind, col_ind].sum()

            if center_distance_min_per_img is None:
                continue

            center_errors_per_img.append(center_distance_min_per_img)

            img_difficulty_num = 0



            for row_id, col_id in zip(row_ind, col_ind):
                diff_label_num_per_row = diff_label[row_id, col_id]
                img_difficulty_num = max(img_difficulty_num, diff_label_num_per_row)
                

            cente_img = np.array(center_errors_per_img).mean()
            if len(center_errors_per_img) > 0:
                center_errors.append(cente_img)

            if img_difficulty_num == 3:
                center_errors_hard.append(cente_img)
                center_errors_outside.append(cente_img)
            elif img_difficulty_num == 2:
                center_errors_medium.append(cente_img)
                center_errors_outside.append(cente_img)
            elif img_difficulty_num == 1:
                center_errors_easy.append(cente_img)
            elif img_difficulty_num == 0:
                center_errors_normal.append(cente_img)
        except KeyError:
            continue

        
    try: 
        ret_dict = {
            'center_errors': np.array(center_errors).mean(),
            'center_errors_normal': np.array(center_errors_normal).mean(),
            'center_errors_easy': np.array(center_errors_easy).mean(),
            'center_errors_medium': np.array(center_errors_medium).mean(),
            'center_errors_hard': np.array(center_errors_hard).mean(),
            'center_errors_outside': np.array(center_errors_outside).mean()
        }
    except RuntimeError:
        print(center_errors)
        print(center_errors_normal)
        print(center_errors_easy)
        print(center_errors_medium)
        print(center_errors_hard)
        print(center_errors_outside)

        ret_dict = {
            'center_errors': np.array(center_errors).mean(),
            'center_errors_normal': np.array(center_errors_normal).mean(),
            'center_errors_easy': np.array(center_errors_easy).mean(),
            'center_errors_medium': np.array(center_errors_medium).mean(),
            'center_errors_hard': np.array(center_errors_hard).mean(),
            'center_errors_outside': np.array(center_errors_outside).mean()
        }


    return ret_dict


def pre_evaluate(dataset, EAD=False):
    imageid2ann = {}
    imageid2size = {}

    img_ids = dataset.coco.getImgIds() # list
    for img_id in img_ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        imageid2ann[img_id] = anns

        img_info = dataset.coco.loadImgs(img_id)[0]
        if EAD:
            imageid2size[img_id] = (img_info['width']*3, img_info['height']*3)
        else:
            imageid2size[img_id] = (img_info['width'], img_info['height'])

    return imageid2ann, imageid2size

def convert_preds(preds, img_ids):
    """
    Args:
        preds = [
            {
                'image_id': 0,
                'category_id': 0,
                'bbox': [0, 0, 10, 10],
                'score': 0.9
            },
            {
                'image_id': 0,
                'category_id': 0,
                'bbox': [0, 0, 10, 10],
                'score': 0.9    
            }
        ]

        img_ids = [0, 1, ...]

    Returns:
        imageid2pred = {
            0: [
                {
                    'category_id': 0,
                    'bbox': [0, 0, 10, 10],
                    'score': 0.9
                },
                {
                    'category_id': 0,
                    'bbox': [0, 0, 10, 10],
                    'score': 0.9
                }
            ]
        }
    """

    imageid2pred = {_id: [] for _id in img_ids}

    tot_preds = len(preds)
    num_print = tot_preds // 50

    for pred_id, pred in enumerate(preds):
        if pred_id % num_print == 0:
            print(f'Processing {pred_id}th/{tot_preds} prediction...')
        imageid2pred[pred['image_id']].append(pred)

    print('convert_preds done')

    return imageid2pred

eval_funs = {
    # 'pred': evaluate_center_error,
    # 'pred_img_wise': evaluate_center_error_img_wise,
    # 'gt': evaluate_center_error_gt,
    # 'gt_img_wise': evaluate_center_error_gt_img_wise,
    'bipartite_mean_se_img_wise': evaluate_center_dis_bipartite,
    # 'bipartite_mean_se_pair_wise': evaluate_center_dis_bipartite_pair_wise,
    # 'bipartite_median_se_img_wise': evaluate_center_dis_bipartite_img_wise_median,
    # 'bipartite_median_se_pair_wise': evaluate_center_dis_bipartite_pair_wise_median,
}

def eval_center_error(eval_fun_name_list, preds, dataset, isface=True, EAD=False):
    print('Preparing data for center error evaluation...')
    imageid2ann, imageid2size = pre_evaluate(dataset, EAD=EAD)
    print('Converting predictions...')
    imageid2pred = convert_preds(preds, dataset.coco.getImgIds())

    # print(imageid2pred)

    center_error_dict = {}
    print('Evaluating center error...')
    for eval_fun_name in eval_fun_name_list:
        print(f'Running {eval_fun_name}...')
        center_error_dict[eval_fun_name] = eval_funs[eval_fun_name](imageid2pred, imageid2ann, imageid2size, isface=isface)

    return center_error_dict

from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np
import json

def _iter_preds_from_json(json_path):
    """Generator yielding detections from the streamed JSON array."""
    with open(json_path, 'r') as f:
        # load as a whole list (fast) – but if super large, switch to ijson (streaming) here
        preds = json.load(f)
    for p in preds:
        yield p

def convert_preds_lean(json_path):
    """
    Build imageid2pred only for images with predictions (saves RAM).
    """
    imageid2pred = defaultdict(list)
    tot = 0
    for i, pred in enumerate(_iter_preds_from_json(json_path)):
        imageid2pred[pred['image_id']].append(pred)
        tot += 1
        # minimal logging
        if (i % max(1, tot // 50)) == 0:
            pass
    return imageid2pred, tot

def pre_evaluate_lean(dataset, img_ids_with_preds, EAD=False):
    """
    Load annotations and sizes only for images with predictions.
    """
    imageid2ann = {}
    imageid2size = {}
    coco = dataset.coco
    for img_id in img_ids_with_preds:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        imageid2ann[img_id] = anns
        img_info = coco.loadImgs(img_id)[0]
        if EAD:
            imageid2size[img_id] = (img_info['width']*3, img_info['height']*3)
        else:
            imageid2size[img_id] = (img_info['width'], img_info['height'])
    return imageid2ann, imageid2size

def eval_center_error_lean(eval_fun_name_list, preds_json_path, dataset, isface=True, EAD=False):
    print('Preparing data for center error evaluation (lean)...')
    imageid2pred, _ = convert_preds_lean(preds_json_path)
    img_ids_with_preds = list(imageid2pred.keys())
    imageid2ann, imageid2size = pre_evaluate_lean(dataset, img_ids_with_preds, EAD=EAD)

    center_error_dict = {}
    for eval_fun_name in eval_fun_name_list:
        center_error_dict[eval_fun_name] = evaluate_center_dis_bipartite(
            imageid2pred, imageid2ann, imageid2size, isface=isface
        )
    return center_error_dict