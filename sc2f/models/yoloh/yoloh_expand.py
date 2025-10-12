import numpy as np
import math
import time
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..neck import build_neck
from ..head.decoupled_head import DecoupledHead
from ..expansion.expansion_net import get_Upsample_layer
from ..basic.conv import Conv
from ..expansion.attn_expander import DetrMLPPredictionHead
from einops import rearrange


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

Has_print_dict = {}

class YOLOH_Expand(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes=20,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 trainable=True,
                 topk=1000, 
                 fig_size=(640, 640),
                 **kwargs):
        super(YOLOH_Expand, self).__init__()
        self.cfg = cfg
        self.device = device
        self.fmp_size = None
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])
        self.EAD = cfg['EAD']

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'], 
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # neck
        self.neck = build_neck(cfg=cfg, 
                               in_dim=bk_dim, 
                               out_dim=cfg['head_dim'])
                                     
        # head
        self.head = DecoupledHead(head=cfg['head'],
                                  head_dim=cfg['head_dim'],
                                  kernel_size=3,
                                  padding=1,
                                  num_classes=num_classes,
                                  trainable=trainable,
                                  num_anchors=self.num_anchors,
                                  act_type=cfg['act_type'])

        # expansion
        assert 'expansion_cfg' in cfg.keys(), 'expansion_cfg is required for YOLOH expansion'
        self.expansion = get_Upsample_layer(**cfg['expansion_cfg'], fig_size=fig_size)
        
        if 'use_before_dilated' in cfg.keys() and cfg['use_before_dilated']:
            self.use_before_dilated = True
        else:
            self.use_before_dilated = False
            
        if 'concat_pred_ori_tokens' in cfg.keys() and cfg['concat_pred_ori_tokens']:
            self.concat_pred_ori_tokens = True
        else:
            self.concat_pred_ori_tokens = False
        
        self.expansion.cls_head = DetrMLPPredictionHead(
            input_dim=cfg['expansion_cfg']['attn']['d_model'],
            hidden_dim=cfg['expansion_cfg']['attn']['d_model'],
            output_dim=self.num_classes * self.num_anchors,
            num_layers=2,
        )
        
        self.expansion.reg_head = DetrMLPPredictionHead(
            input_dim=cfg['expansion_cfg']['attn']['d_model'],
            hidden_dim=cfg['expansion_cfg']['attn']['d_model'],
            output_dim=4 * self.num_anchors,
            num_layers=2,
        )
        
        if 'use_objness' in cfg.keys() and cfg['use_objness']:
            self.expansion.obj_head = DetrMLPPredictionHead(
                input_dim=cfg['expansion_cfg']['attn']['d_model'],
                hidden_dim=cfg['expansion_cfg']['attn']['d_model'],
                output_dim=1 * self.num_anchors,
                num_layers=2,
            )
            
        if 'use_objfest' in cfg.keys() and cfg['use_objfest']:
            self.expansion.obj_fest = DetrMLPPredictionHead(
                input_dim=cfg['expansion_cfg']['attn']['d_model'],
                hidden_dim=cfg['expansion_cfg']['attn']['d_model'],
                output_dim=cfg['expansion_cfg']['attn']['d_model'],
                num_layers=2,
            )
            


    def generate_anchors(self, fmp_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # check anchor boxes
        if self.fmp_size is not None and self.fmp_size == fmp_size:
            return self.anchor_boxes
        else:
            # generate grid cells
            fmp_h, fmp_w = fmp_size
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
            # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
            anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
            anchor_xy *= self.stride

            # if self.EAD:
            #     # In EAD, the feature map is in the center of the expanded area, therefore, the xy should be doubled
            #     anchor_xy = anchor_xy + torch.tensor([fmp_w, fmp_h], device=self.device)[None, None, :]

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

            self.anchor_boxes = anchor_boxes
            self.fmp_size = fmp_size

            return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            pred_reg: (List[tensor]) [B, M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                      max=self.cfg['ctr_clamp'],
                                      min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
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
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def feature(self, x, use_before_dilated=False):
        # backbone
        x = self.backbone(x)

        # neck
        x, before_dilated = self.neck(x)

        if use_before_dilated:
            return x, before_dilated
        
        return  x

    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]

        if self.EAD:
            img_h, img_w = img_h * 3, img_w * 3
        # backbone
        x = self.backbone(x)

        # neck
        x, before_dilated = self.neck(x)

            
        H, W = x.shape[2:]

        # head

        expanded_H, expanded_W = H*3, W*3

        # head
        cls_pred, reg_pred = self.head(x) #[B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        
        if self.use_before_dilated:
            expansion_output  = self.expansion(before_dilated, trainable=False, concatenate=self.concat_pred_ori_tokens) # [B, C, H, W] -> [B, C, 3H, 3W]
        else:
            expansion_output  = self.expansion(x, trainable=False, concatenate=self.concat_pred_ori_tokens)
            
        if isinstance(expansion_output, tuple):
            expanded_x, expanded_reassign_coordinates_on_device = expansion_output
        elif isinstance(expansion_output, dict):
            expanded_x = expansion_output['expanded_features']
            expanded_reassign_coordinates_on_device = expansion_output['expanded_reassign_coordinates_on_device']
        else:
            raise ValueError('expansion_output should be a tuple or a dict, but got {}'.format(type(expansion_output)))
        
        expanded_cls = self.expansion.cls_head(expanded_x) # [B, Num_tokens_to_expand, KA*C]
            
        if hasattr(self.expansion, 'obj_fest'):
            obj_fest = self.expansion.obj_fest(expanded_x) # [B, Num_tokens_to_expand, D] D is the hidden dimension
        else:
            obj_fest = expanded_x
        
        expanded_reg = self.expansion.reg_head(obj_fest) # [B, Num_tokens_to_expand, KA*4]
        
        if hasattr(self.expansion, 'obj_head'):
            expanded_obj = self.expansion.obj_head(obj_fest) # [B, Num_tokens_to_expand, KA*1]
            expanded_cls = rearrange(expanded_cls, 'b n (k c) -> b k c n', k=self.num_anchors)
            expanded_obj = rearrange(expanded_obj, 'b n (k c) -> b k c n', k=self.num_anchors)
            expanded_cls = expanded_cls + expanded_obj - torch.log(1. + torch.clamp(expanded_cls.exp(), max=1e8) + torch.clamp(expanded_obj.exp(), max=1e8))
            expanded_cls = rearrange(expanded_cls, 'b k c n -> b n (k c)').contiguous()
        
        if 'expanded_cls' not in Has_print_dict:
            print('expanded_cls:', expanded_cls.shape, 'expanded_cls.dtype:', expanded_cls.dtype)
            print('expanded_reg:', expanded_reg.shape, 'expanded_reg.dtype:', expanded_reg.dtype)
            Has_print_dict['expanded_cls'] = 1
            
        # expanded token index only for expanded parts
        if not hasattr(self, 'expanded_token_index'):
            reduce_factor = self.expansion.reduce_factor
            expanded_token_index = torch.zeros((3*H, 3*W), dtype=torch.int64, device=self.device)
            index_ = 0
            assert 3 * H % reduce_factor == 0 and 3 * W % reduce_factor == 0, '3H and 3W should be divisible by reduce_factor'
            for i in range(3*H // reduce_factor):
                for j in range(3*W // reduce_factor):
                    if (i * reduce_factor >= H) and (j * reduce_factor >= W) and (i * reduce_factor < 2*H) and (j * reduce_factor < 2*W):
                        expanded_token_index[i, j] = -1
                        continue
                    
                    for k in range(reduce_factor):
                        for l in range(reduce_factor):
                            if (i * reduce_factor + k >= H) and (j * reduce_factor + l >= W) and (i * reduce_factor + k < 2*H) and (j * reduce_factor + l < 2*W):
                                continue
                            expanded_token_index[i * reduce_factor + k, j * reduce_factor + l] = index_
                    index_ += 1
            # [3H, 3W] -> [B, 3H, 3W, KA*C] repeat
            # print(f'expanded_token_index: {expanded_token_index}')
            expanded_token_index = expanded_token_index[None, :, :, None].repeat(expanded_cls.shape[0], 1, 1, self.num_anchors)
            self.expanded_token_index = expanded_token_index.clone()
            expanded_token_index = rearrange(expanded_token_index, 'b h w k -> b (h w k)') # [B, HW*KA]
        elif isinstance(expansion_output, dict) and 'token_idx_manager' in expansion_output.keys():
            # use the token index manager to get the expanded token index
            token_idx_manager = expansion_output['token_idx_manager']
            token_id_map = token_idx_manager.token_id_map # [B, H, W]
            token_id_map = token_id_map.repeat(1, 1, 1, self.num_anchors) # [B, H, W, KA]
            expanded_token_index = rearrange(token_id_map, 'b h w k -> b (h w k)') # [B, HW*KA]
        else:
            expanded_token_index = rearrange(self.expanded_token_index, 'b h w k -> b (h w k)') # [B, HW*KA]
        
        cls_pred = rearrange(cls_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
        reg_pred = rearrange(reg_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
        
        # expanded_cls[:, H:2*H, W:2*W] = cls_pred
        # expanded_reg[:, H:2*H, W:2*W] = reg_pred
        cls_pred = torch.nn.functional.pad(cls_pred, (H, H, W, W), value=0, )
        reg_pred = torch.nn.functional.pad(reg_pred, (H, H, W, W), value=0, )
        cls_pred = rearrange(cls_pred, 'b c h w -> b h w c')
        reg_pred = rearrange(reg_pred, 'b c h w -> b h w c')
        
        if cls_pred.dtype != expanded_cls.dtype:
            expanded_cls = expanded_cls.to(cls_pred.dtype)
        
        if expanded_reassign_coordinates_on_device.dim() == 3:
            batch_idx = torch.arange(0, cls_pred.shape[0], device=cls_pred.device).unsqueeze(1)
            cls_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_cls
            reg_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_reg
        else:
            cls_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_cls
            reg_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_reg
        
        cls_pred = rearrange(cls_pred, 'b h w (k c) -> b (h w k) c', h=expanded_H, w=expanded_W, k=self.num_anchors)
        reg_pred = rearrange(reg_pred, 'b h w (k c) -> b (h w k) c', h=expanded_H, w=expanded_W, k=self.num_anchors)
        # decode box
        anchor_boxes = self.generate_anchors(fmp_size=[expanded_H, expanded_W]) # [M, 4]
        # scores
        scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

        # topk
        if scores.shape[0] == 1:
            scores = scores.squeeze(0)
            labels = labels.squeeze(0)
            reg_pred = reg_pred.squeeze(0)
            expanded_token_index = expanded_token_index.squeeze(0)
            anchor_boxes = anchor_boxes
        
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]
            expanded_token_index = expanded_token_index[indices]

        # decode box
        bboxes = self.decode_boxes(anchor_boxes[None], reg_pred[None])[0] # [N, 4]

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()
        expanded_token_index = expanded_token_index.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]
        expanded_token_index = expanded_token_index[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int64)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        expanded_token_index = expanded_token_index[keep]

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)
        
        return_list = {
            'bboxes': bboxes,
            'scores': scores,
            'labels': labels,
            'token_idx': expanded_token_index,
            'token_idx_manager': expansion_output['token_idx_manager'] if isinstance(expansion_output, dict) and 'token_idx_manager' in expansion_output.keys() else None
        }

        # return bboxes, scores, labels
        return return_list
    
    def pre_box(self, x, mask=None):
        cls_pred, reg_pred = self.head(x)
        H, W = x.shape[2:]

        # decode box
        anchor_boxes = self.generate_anchors(fmp_size=[H, W]) # [M, 4]
        box_pred = self.decode_boxes(anchor_boxes[None], reg_pred) # [B, M, 4]
        
        if mask is not None:
            # [B, H, W]
            mask = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
            # [B, H, W] -> [B, HW]
            mask = mask.flatten(1)
            # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
            mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()
            
        outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "mask": mask}
        
        return outputs

    @torch.no_grad()
    def heatmap(self, x, mask=None):
        img_h, img_w = x.shape[2:]

        if self.EAD:
            img_h, img_w = img_h * 3, img_w * 3
        # backbone
        x = self.backbone(x)

        # neck
        x, before_dilated = self.neck(x)

            
        H, W = x.shape[2:]

        # head

        expanded_H, expanded_W = H*3, W*3

        # head
        cls_pred, reg_pred = self.head(x) #[B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        
        if self.use_before_dilated:
            expansion_output  = self.expansion(before_dilated, trainable=False, concatenate=self.concat_pred_ori_tokens) # [B, C, H, W] -> [B, C, 3H, 3W]
        else:
            expansion_output  = self.expansion(x, trainable=False, concatenate=self.concat_pred_ori_tokens)
            
        if isinstance(expansion_output, tuple):
            expanded_x, expanded_reassign_coordinates_on_device = expansion_output
        elif isinstance(expansion_output, dict):
            expanded_x = expansion_output['expanded_features']
            expanded_reassign_coordinates_on_device = expansion_output['expanded_reassign_coordinates_on_device']
        else:
            raise ValueError('expansion_output should be a tuple or a dict, but got {}'.format(type(expansion_output)))
        
        expanded_cls = self.expansion.cls_head(expanded_x) # [B, Num_tokens_to_expand, KA*C]
            
        if hasattr(self.expansion, 'obj_fest'):
            obj_fest = self.expansion.obj_fest(expanded_x) # [B, Num_tokens_to_expand, D] D is the hidden dimension
        else:
            obj_fest = expanded_x
        
        expanded_reg = self.expansion.reg_head(obj_fest) # [B, Num_tokens_to_expand, KA*4]
        
        if hasattr(self.expansion, 'obj_head'):
            expanded_obj = self.expansion.obj_head(obj_fest) # [B, Num_tokens_to_expand, KA*1]
            expanded_cls = rearrange(expanded_cls, 'b n (k c) -> b k c n', k=self.num_anchors)
            expanded_obj = rearrange(expanded_obj, 'b n (k c) -> b k c n', k=self.num_anchors)
            expanded_cls = expanded_cls + expanded_obj - torch.log(1. + torch.clamp(expanded_cls.exp(), max=1e8) + torch.clamp(expanded_obj.exp(), max=1e8))
            expanded_cls = rearrange(expanded_cls, 'b k c n -> b n (k c)').contiguous()
        
        if 'expanded_cls' not in Has_print_dict:
            print('expanded_cls:', expanded_cls.shape, 'expanded_cls.dtype:', expanded_cls.dtype)
            print('expanded_reg:', expanded_reg.shape, 'expanded_reg.dtype:', expanded_reg.dtype)
            Has_print_dict['expanded_cls'] = 1
            
        # expanded token index only for expanded parts
        if not hasattr(self, 'expanded_token_index'):
            reduce_factor = self.expansion.reduce_factor
            expanded_token_index = torch.zeros((3*H, 3*W), dtype=torch.int64, device=self.device)
            index_ = 0
            assert 3 * H % reduce_factor == 0 and 3 * W % reduce_factor == 0, '3H and 3W should be divisible by reduce_factor'
            for i in range(3*H // reduce_factor):
                for j in range(3*W // reduce_factor):
                    if (i * reduce_factor >= H) and (j * reduce_factor >= W) and (i * reduce_factor < 2*H) and (j * reduce_factor < 2*W):
                        expanded_token_index[i, j] = -1
                        continue
                    
                    for k in range(reduce_factor):
                        for l in range(reduce_factor):
                            if (i * reduce_factor + k >= H) and (j * reduce_factor + l >= W) and (i * reduce_factor + k < 2*H) and (j * reduce_factor + l < 2*W):
                                continue
                            expanded_token_index[i * reduce_factor + k, j * reduce_factor + l] = index_
                    index_ += 1
            # [3H, 3W] -> [B, 3H, 3W, KA*C] repeat
            # print(f'expanded_token_index: {expanded_token_index}')
            expanded_token_index = expanded_token_index[None, :, :, None].repeat(expanded_cls.shape[0], 1, 1, self.num_anchors)
            self.expanded_token_index = expanded_token_index.clone()
            expanded_token_index = rearrange(expanded_token_index, 'b h w k -> b (h w k)') # [B, HW*KA]
        elif isinstance(expansion_output, dict) and 'token_idx_manager' in expansion_output.keys():
            # use the token index manager to get the expanded token index
            token_idx_manager = expansion_output['token_idx_manager']
            token_id_map = token_idx_manager.token_id_map # [B, H, W]
            token_id_map = token_id_map.repeat(1, 1, 1, self.num_anchors) # [B, H, W, KA]
            expanded_token_index = rearrange(token_id_map, 'b h w k -> b (h w k)') # [B, HW*KA]
        else:
            expanded_token_index = rearrange(self.expanded_token_index, 'b h w k -> b (h w k)') # [B, HW*KA]
        
        cls_pred = rearrange(cls_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
        # reg_pred = rearrange(reg_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
        
        # expanded_cls[:, H:2*H, W:2*W] = cls_pred
        # expanded_reg[:, H:2*H, W:2*W] = reg_pred
        cls_pred = torch.nn.functional.pad(cls_pred, (H, H, W, W), value=0, )
        # reg_pred = torch.nn.functional.pad(reg_pred, (H, H, W, W), value=0, )
        cls_pred = rearrange(cls_pred, 'b c h w -> b h w c')
        # reg_pred = rearrange(reg_pred, 'b c h w -> b h w c')
        
        if cls_pred.dtype != expanded_cls.dtype:
            expanded_cls = expanded_cls.to(cls_pred.dtype)
        
        if expanded_reassign_coordinates_on_device.dim() == 3:
            batch_idx = torch.arange(0, cls_pred.shape[0], device=cls_pred.device).unsqueeze(1)
            cls_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_cls
            # reg_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_reg
        else:
            cls_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_cls
            # reg_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_reg
        
        cls_pred = rearrange(cls_pred, 'b h w (k c) -> b (h w) c k', h=expanded_H, w=expanded_W, k=self.num_anchors).max(dim=-1)[0]
        # reg_pred = rearrange(reg_pred, 'b h w (k c) -> b (h w k) c', h=expanded_H, w=expanded_W, k=self.num_anchors)
        
        return cls_pred

    def forward(self, x, mask=None):

        global Has_print_dict

        if torch.isnan(x).any():
            print('Warning: input contains NaN')
            raise ValueError('NaN in input')

        if 'input' not in Has_print_dict:
            print('input:', x.shape)
            Has_print_dict['input'] = 1

        if not self.trainable:
            # print('Warning: Inference Mode')
            return self.inference_single_image(x)
        else:
            # backbone
            x = self.backbone(x)

            if torch.isnan(x[0]).any() or torch.isnan(x[1]).any() or torch.isnan(x[2]).any():
                print('Warning: backbone contains NaN')
                raise ValueError('NaN in backbone')

            

            if 'backbone' not in Has_print_dict:
                print('backbone(len):', len(x))
                for i in range(len(x)):
                    print(f'backbone-{i}:', x[i].shape)
                Has_print_dict['backbone'] = 1

            # neck
            x, before_dilated = self.neck(x)

            if torch.isnan(x).any():
                print('Warning: neck contains NaN')
                raise ValueError('NaN in neck')

            if 'neck' not in Has_print_dict:
                print('neck:', x.shape, 'neck.dtype:', x.dtype)
                Has_print_dict['neck'] = 1


            H, W = x.shape[2:]

            expanded_H, expanded_W = H*3, W*3

            # head
            cls_pred, reg_pred = self.head(x) #[B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
            
            
            # expanded_x = self.expansion(x) # [B, C, H, W] -> [B, C, 3H, 3W]
            
            # When set concatenate to False, the output will be expanded tokens only, without the original tokens.
            if self.use_before_dilated:
                expansion_output = self.expansion(before_dilated, trainable=False, concatenate=self.concat_pred_ori_tokens) # [B, C, H, W] -> [B, C, 3H, 3W]
            else:
                expansion_output = self.expansion(x, trainable=False, concatenate=self.concat_pred_ori_tokens)
            # expanded_x, expanded_reassign_coordinates_on_device = self.expansion(x, concatenate=False) # [B, C, H, W] -> [B, Num_tokens_to_expand, C]
            
            if isinstance(expansion_output, tuple):
                expanded_x, expanded_reassign_coordinates_on_device = expansion_output
            elif isinstance(expansion_output, dict):
                expanded_x = expansion_output['expanded_features']
                expanded_reassign_coordinates_on_device = expansion_output['expanded_reassign_coordinates_on_device']
                token_scores = expansion_output['token_scores']
                token_idx_manager = expansion_output['token_idx_manager']
                # print('[DEBUG] token_scores.shape:', token_scores.shape)
            else:
                raise ValueError('expansion_output should be a tuple or a dict, but got {}'.format(type(expansion_output)))
            
            if 'expanded_x' not in Has_print_dict:
                print('expanded_x:', expanded_x.shape, 'expanded_x.dtype:', expanded_x.dtype)
                Has_print_dict['expanded_x'] = 1
            
            # expanded_x = rearrange(expanded_x, 'b c h w -> b h w c')
            
            # expanded_cls = self.expansion.cls_head(expanded_x) # [B, 3H, 3W, KA*C]
            # expanded_reg = self.expansion.reg_head(expanded_x) # [B, 3H, 3W, KA*4]
            
            expanded_cls = self.expansion.cls_head(expanded_x) # [B, Num_tokens_to_expand, KA*C]
            
            if hasattr(self.expansion, 'obj_fest'):
                obj_fest = self.expansion.obj_fest(expanded_x) # [B, Num_tokens_to_expand, D] D is the hidden dimension
            else:
                obj_fest = expanded_x
            
            expanded_reg = self.expansion.reg_head(obj_fest) # [B, Num_tokens_to_expand, KA*4]
            
            if hasattr(self.expansion, 'obj_head'):
                expanded_obj = self.expansion.obj_head(obj_fest) # [B, Num_tokens_to_expand, KA*1]
                expanded_cls = rearrange(expanded_cls, 'b n (k c) -> b k c n', k=self.num_anchors)
                expanded_obj = rearrange(expanded_obj, 'b n (k c) -> b k c n', k=self.num_anchors)
                expanded_cls = expanded_cls + expanded_obj - torch.log(1. + torch.clamp(expanded_cls.exp(), max=1e8) + torch.clamp(expanded_obj.exp(), max=1e8))
                expanded_cls = rearrange(expanded_cls, 'b k c n -> b n (k c)').contiguous()

                
            
            if 'expanded_cls' not in Has_print_dict:
                print('expanded_cls:', expanded_cls.shape, 'expanded_cls.dtype:', expanded_cls.dtype)
                print('expanded_reg:', expanded_reg.shape, 'expanded_reg.dtype:', expanded_reg.dtype)
                Has_print_dict['expanded_cls'] = 1
            
            cls_pred = rearrange(cls_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
            reg_pred = rearrange(reg_pred, 'b (h w k) c -> b (k c) h w', h=H, w=W, k=self.num_anchors)
            
            # expanded_cls[:, H:2*H, W:2*W] = cls_pred
            # expanded_reg[:, H:2*H, W:2*W] = reg_pred
            cls_pred = torch.nn.functional.pad(cls_pred, (H, H, W, W), value=0, )
            reg_pred = torch.nn.functional.pad(reg_pred, (H, H, W, W), value=0, )
            cls_pred = rearrange(cls_pred, 'b c h w -> b h w c')
            reg_pred = rearrange(reg_pred, 'b c h w -> b h w c')
            
            if 'head' not in Has_print_dict:
                print('head:', cls_pred.shape, reg_pred.shape, 'cls_pred.dtype:', cls_pred.dtype, 'reg_pred.dtype:', reg_pred.dtype)
                Has_print_dict['head'] = 1
            
            if cls_pred.dtype != expanded_cls.dtype:
                expanded_cls = expanded_cls.to(cls_pred.dtype)
            
            if expanded_reassign_coordinates_on_device.dim() == 3:
                batch_idx = torch.arange(0, cls_pred.shape[0], device=cls_pred.device).unsqueeze(1)
                cls_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_cls
                reg_pred[batch_idx, expanded_reassign_coordinates_on_device[..., 0], expanded_reassign_coordinates_on_device[..., 1], :] = expanded_reg
            else:
                cls_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_cls
                reg_pred[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_reg

            cls_pred = rearrange(cls_pred, 'b h w (k c) -> b (h w k) c', h=expanded_H, w=expanded_W, k=self.num_anchors)
            reg_pred = rearrange(reg_pred, 'b h w (k c) -> b (h w k) c', h=expanded_H, w=expanded_W, k=self.num_anchors)
            
            
            if torch.isnan(expanded_cls).any():
                print('Warning: expanded_cls contains NaN')
                raise ValueError('NaN in expanded_cls')
            if torch.isnan(expanded_reg).any():
                print('Warning: expanded_reg contains NaN')
                raise ValueError('NaN in expanded_reg')

            if torch.isnan(cls_pred).any():
                print('Warning: cls_pred contains NaN')
                raise ValueError('NaN in cls_pred')
            if torch.isnan(reg_pred).any():
                print('Warning: reg_pred contains NaN')
                raise ValueError('NaN in reg_pred')


            # decode box
            anchor_boxes = self.generate_anchors(fmp_size=[expanded_H, expanded_W]) # [M, 4], M = Expanded_H * Expanded_W * num_anchors
            box_pred = self.decode_boxes(anchor_boxes[None], reg_pred) # [B, M, 4]

            if 'anchor_boxes' not in Has_print_dict:
                print('anchor_boxes:', anchor_boxes.shape)
                print('anchor_boxes:', anchor_boxes[:5])
                Has_print_dict['anchor_boxes'] = 1

            if 'box_pred' not in Has_print_dict:
                print('box_pred:', box_pred.shape)
                print('box_pred:', box_pred[0, :5])
                Has_print_dict['box_pred'] = 1
            
            if mask is not None:
                # [B, H, W]
                mask = torch.nn.functional.interpolate(mask[None], size=[expanded_H, expanded_W]).bool()[0]
                # [B, H, W] -> [B, HW]
                mask = mask.flatten(1)
                # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
                mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()
                
            # print('[DEBUG] cls_pred:', cls_pred.shape, 'reg_pred:', reg_pred.shape, 'box_pred:', box_pred.shape, 'mask:', mask.shape if mask is not None else None)

            outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "mask": mask,
                       'token_idx_manager': token_idx_manager if 'token_idx_manager' in locals() else None,
                       'token_scores': token_scores if 'token_scores' in locals() else None,
                   }


            return outputs
