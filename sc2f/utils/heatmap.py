import numpy as np
import math
import torch


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    # print(f'gaussian: {gaussian.shape}', gaussian)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[:2]

    # print(f'height: {height}, width: {width}, x: {x}, y: {y}, radius: {radius}, center: {center}')
        
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    # print(f'left: {left}, right: {right}, top: {top}, bottom: {bottom}')

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]

    # print(f'masked_heatmap: {masked_heatmap.shape}', masked_heatmap)

    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    # print(f'masked_gaussian: {masked_gaussian.shape}', masked_gaussian)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # print(f'masked_heatmap: {masked_heatmap.shape}', masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def convert_bbox_heatmap_per_img_per_category(img, bboxes, category_id):
    """
    The data should be the image after transform
    The target should be the target after transform

    img: tensor, shape (h, w, 3)
    """

    h, w = img.shape[:2]
    hm = np.zeros((h, w, 1), dtype=np.float32)

    for bbox in bboxes:
        if bbox[4] != category_id:
            continue
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        print(f'radius: {radius}')
        # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        print(f'ct_int: {ct_int}')
        draw_umich_gaussian(hm[:,:,category_id], ct_int, radius)

    return hm

def convert_bbox_heatmap_per_img(img, bboxes, num_classes):
    """
    The data should be the image after transform
    The target should be the target after transform

    img: tensor, shape (h, w, 3)
    """

    h, w,  = img.shape[:2]
    hm = np.zeros((h, w, num_classes), dtype=np.float32)

    for bbox in bboxes:
        if bbox.shape[0] == 5:
            bbox_category_id = bbox[4]
            if 'int' not in str(type(bbox_category_id)):
                bbox_category_id = int(bbox_category_id)
        else:
            print('Unknown category id')
            continue

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        print(f'radius: {radius}')
        # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        print(f'ct_int: {ct_int}')
        draw_umich_gaussian(hm[:,:,bbox_category_id], ct_int, radius)

    return hm

# def convert_bbox_heatmap_per_img_tensor(img, bboxes, num_classes):
#     """
#     The data should be the image after transform
#     The target should be the target after transform

#     img: tensor, shape (h, w, 3)
#     """

#     c, h, w  = img.shape
#     hm = np.zeros((num_classes, h, w), dtype=np.float32)

#     for bbox in bboxes:
#         if bbox.shape[0] == 5:
#             bbox_category_id = bbox[4]
#             if 'int' not in str(type(bbox_category_id)):
#                 bbox_category_id = int(bbox_category_id)
#         else:
#             print('Unknown category id')
#             continue

#         radius = gaussian_radius((math.ceil(h), math.ceil(w)))
#         radius = max(0, int(radius))
#         # print(f'radius: {radius}')
#         # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
#         ct = np.array(
#         [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
#         ct_int = ct.astype(np.int32)
#         # print(f'ct_int: {ct_int}')
#         draw_umich_gaussian(hm[bbox_category_id, :, :], ct_int, radius)

#     return torch.tensor(hm)

def convert_bbox_heatmap_per_img_tensor(img, bboxes, num_classes, down_ratio=1, radius_down_ratio=1):
    """
    The data should be the image after transform
    The target should be the target after transform

    img: tensor, shape (h, w, 3)
    """

    c, h, w  = img.shape

    h, w = h // down_ratio, w // down_ratio

    hm = np.zeros((num_classes, h, w), dtype=np.float32)

    for bbox in bboxes:
        if bbox.shape[0] == 5:
            bbox_category_id = bbox[4]
            if 'int' not in str(type(bbox_category_id)):
                bbox_category_id = int(bbox_category_id)
        else:
            print('Unknown category id')
            continue

        radius = gaussian_radius((math.ceil(h), math.ceil(w))) * radius_down_ratio
        radius = max(0, int(radius))
        # print(f'radius: {radius}')
        # radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
        [(bbox[0] // down_ratio + bbox[2] // down_ratio) / 2, (bbox[1] // down_ratio + bbox[3] // down_ratio) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        # print(f'ct_int: {ct_int}')
        draw_umich_gaussian(hm[bbox_category_id, :, :], ct_int, radius)

    return torch.tensor(hm)

def pad_tensor_image_3(image):
    h, w = image.shape[1], image.shape[2]
    pad = (w, w, h, h)
    image = torch.nn.functional.pad(image, pad, value=0)
    return image