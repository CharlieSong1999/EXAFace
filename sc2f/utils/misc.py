import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def nms(dets, scores, nms_thresh=0.4):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 0]  #xmin
    y1 = dets[:, 1]  #ymin
    x2 = dets[:, 2]  #xmax
    y2 = dets[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


from utils.heatmap import convert_bbox_heatmap_per_img_tensor

def pad_tensor_image_3(image):
    h, w = image.shape[1], image.shape[2]
    pad = (w, w, h, h)
    image = torch.nn.functional.pad(image, pad, value=0)
    return image

class HeatmapCollector():
    def __init__(self, num_classes, down_ratio=16, radius_down_ratio=1, *args, **kwargs):
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.radius_down_ratio = radius_down_ratio

    def __call__(self, batch):
        batch_size = len(batch)
        images = []
        targets = []
        masks = []
        heatmaps = []
        for bid in range(batch_size):
            images.append(batch[bid][0])
            targets.append(batch[bid][1])
            masks.append(batch[bid][2])
            boxes_with_labels = torch.cat((batch[bid][1]['boxes'], rearrange(batch[bid][1]['labels'], 'c -> c 1')), dim=1)
            pad_img = pad_tensor_image_3(batch[bid][0])
            heatmaps.append(convert_bbox_heatmap_per_img_tensor(pad_img, boxes_with_labels, self.num_classes, self.down_ratio, self.radius_down_ratio))

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        heatmaps = torch.stack(heatmaps, 0)

        return images, targets, masks, heatmaps

class CollateFunc(object):

    def __init__(self, use_hm_loss=False, dense_guidance=False, *args, **kwargs):
        self.use_hm_loss = use_hm_loss
        self.dense_guidance = dense_guidance
        
        if use_hm_loss:
            self.collect_heatmap = HeatmapCollector(*args, **kwargs)

    def __call__(self, batch):
        
        if self.use_hm_loss:
            return self.collect_heatmap(batch)
        
        targets = []
        images = []
        masks = []
        
        if self.dense_guidance:
            source_imgs = []
            source_masks = []
        

        for sample in batch:
            image = sample[0]
            target = sample[1]
            mask = sample[2]

            images.append(image)
            targets.append(target)
            masks.append(mask)
            
            if self.dense_guidance:
                source_img = sample[3]
                source_mask = sample[4]
                source_imgs.append(source_img)
                source_masks.append(source_mask)

        images = torch.stack(images, 0) # [B, C, H, W]
        masks = torch.stack(masks, 0)   # [B, H, W]
        
        if self.dense_guidance:
            source_imgs = torch.stack(source_imgs, 0)
            source_masks = torch.stack(source_masks, 0)
            return images, targets, masks, source_imgs, source_masks

        return images, targets, masks


# test time augmentation(TTA)
class TestTimeAugmentation(object):
    def __init__(self, num_classes=80, nms_thresh=0.4, scale_range=[320, 640, 32]):
        self.nms = nms
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(scale_range[0], scale_range[1]+1, scale_range[2])
        
    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            if x.size(-1) == s and x.size(-2) == s:
                x_scale = x
            else:
                x_scale =torch.nn.functional.interpolate(
                                        input=x, 
                                        size=(s, s), 
                                        mode='bilinear', 
                                        align_corners=False)
            model.set_grid(s)
            bboxes, scores, labels = model(x_scale)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            x_flip = torch.flip(x_scale, [-1])
            bboxes, scores, labels = model(x_flip)
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels
