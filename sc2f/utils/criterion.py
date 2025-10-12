import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import *
from utils.matcher import UniformMatcher
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from utils.expand import generate_weight_crop_uncrop_gt


class SigmoidFocalWithLogitsLoss(nn.Module):
    """
        focal loss with sigmoid
    """
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(SigmoidFocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class TokenScoreLoss(nn.Module):
    """
        Token score loss, used to supervise the token scores.
        The loss is calculated as the mean squared error between the predicted token scores and the target token scores.
    """
    def __init__(self, reduction='mean', num_anchor=6, **kwargs):
        super(TokenScoreLoss, self).__init__()
        self.num_anchor = num_anchor
        # self.

    def forward(self, pred_scores, matched_indexes, token_idx_manager):
        """
            pred_scores: (Tensor) [B, M]
            matched_indexes: (List) [B, 2]
            where matched_indexes[i] = (src_idx, tgt_idx) means the i-th predicted token is matched with the tgt_idx-th target token.
            The pred_scores are the scores of the predicted tokens, and the target scores are the scores of the matched target tokens.
        """
        
        loss = torch.tensor(0.0, device=pred_scores.device)
        
        for b in range(len(matched_indexes)):
            scores = pred_scores[b]
            # scores: [M]
            src_idx, tgt_idx = matched_indexes[b]
            # src_idx: [M] but not token index
            box_id_to_token_id = token_idx_manager.box_id_to_token_id
            
            # print('src_idx.max()', src_idx.max())
            # print('src_idx.min()', src_idx.min())
            # print('src_idx.shape', src_idx.shape)
            # print('scores.shape', scores.shape)
            
            box_id_to_token_id = box_id_to_token_id.repeat_interleave(self.num_anchor, dim=0)
            # print('box_id_to_token_id.shape', box_id_to_token_id.shape)
            
            # box_id_to_token_id: [H
            token_ids = box_id_to_token_id[src_idx] # H_feat * 3 * W_feat * 3 - H_feat * W_feat, 2
            
            # print('[DEBUG] token_ids:', token_ids)
            
            tgt_scores = torch.zeros_like(scores)
            
            for granularity in token_idx_manager.token_id_map_per_granularity.keys():
                token_id_map = token_idx_manager.token_id_map_per_granularity[granularity][b]
                
                # print('[DEBUG] token_id_map.shape:', token_id_map.shape)
                
                tgt_token_ids = token_id_map[token_ids[:, 0], token_ids[:, 1]] 
                # Remove those with value -1
                tgt_token_ids = tgt_token_ids[tgt_token_ids != -1]
                # token_id_map: [H_feat * 3, W_feat * 3]
                tgt_scores[tgt_token_ids] = 1.0
            
            loss += F.binary_cross_entropy_with_logits(
                input=scores,
                target=tgt_scores,
                reduction='mean'
            )

        loss = loss / len(matched_indexes)

        return loss


class Criterion(nn.Module):
    def __init__(self, 
                cfg, 
                device, 
                alpha=0.25,
                gamma=2.0,
                loss_cls_weight=1.0, 
                loss_reg_weight=1.0,
                loss_hm_weight=1.0, 
                loss_mse_weight=1.0,
                dense_guidence=False,
                weight_map=None,
                num_classes=80,
                num_anchor=6, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.matcher = UniformMatcher(cfg['topk'])
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_hm_weight = loss_hm_weight
        self.loss_mse_weight = loss_mse_weight
        self.dense_guidence = dense_guidence
        self.weight_map = weight_map
        self.num_anchor = num_anchor
        
        if 're-weight-type' in cfg:
            self.re_weight_type = cfg['re-weight-type']
        else:
            self.re_weight_type = 'none'
            
        if 'expansion_cfg' in cfg.keys() and 'attn' in cfg['expansion_cfg'].keys() and 'coarse_to_fine' in cfg['expansion_cfg']['attn']:
            self.coarse_to_fine = cfg['expansion_cfg']['attn']['coarse_to_fine']
            self.loss_token_weight = cfg.get('loss_token_weight', 1.0)
            self.tokenscore_loss = TokenScoreLoss(reduction='mean', num_anchor=num_anchor)
            print('Using coarse to fine token score loss')
        else:
            self.coarse_to_fine = False

        self.cls_loss_f = SigmoidFocalWithLogitsLoss(reduction='none', gamma=gamma, alpha=alpha)


    def loss_labels(self, pred_cls, tgt_cls, num_boxes, weights=None):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = self.cls_loss_f(pred_cls, tgt_cls)

        if weights is not None:
            # print('loss_cls.shape', loss_cls.shape)
            # print('weights.shape', weights.shape)
            loss_cls = loss_cls * weights[:, None]

        return loss_cls.sum() / num_boxes


    def loss_bboxes(self, pred_box, tgt_box, num_boxes):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        # giou
        pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]

        # giou loss
        loss_reg = 1. - torch.diag(pred_giou)

        if self.weight_map is not None and self.re_weight_type != 'none':
            if self.re_weight_type == 'target':
                tgt_box_center = (tgt_box[:, :2] + tgt_box[:, 2:]) // 2
                tgt_weights = self.weight_map[tgt_box_center[:, 1].long(), tgt_box_center[:, 0].long()]
                
                tgt_weights = tgt_weights.to(self.device)

                # print('tgt_weights.shape', tgt_weights.shape)
                # print('loss_reg.shape', loss_reg.shape)
                loss_reg = loss_reg * tgt_weights
            elif self.re_weight_type == 'value':
                pass
            else:
                raise ValueError('Unknown re-weight type: {}'.format(self.re_weight_type))

        return loss_reg.sum() / num_boxes
    
    def loss_fest(self, pred_fest, tgt_fest, mask):
        
        loss_mse = F.mse_loss(pred_fest[mask], tgt_fest[mask])
        
        return loss_mse


    def forward(self,
                outputs, 
                targets, 
                anchor_boxes=None,
                heatmap=None,):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchor_boxes: (Tensor) [M, 4]
        """
        
        loss_dict = {}
        
        pred_box = outputs['pred_box']
        pred_cls = outputs['pred_cls'].reshape(-1, self.num_classes)
        pred_box_copy = pred_box.detach().clone().cpu()
        anchor_boxes_copy = anchor_boxes.clone().cpu()
        # rescale tgt boxes
        B = len(targets)
        indices = self.matcher(pred_box_copy, anchor_boxes_copy, targets)
        
        if self.coarse_to_fine:
            # print('[DEBUG] Using coarse to fine token score loss')
            # generate token_idx_manager
            token_idx_manager = outputs['token_idx_manager']
            # get the token scores
            pred_token_scores = outputs['token_scores'] # [B, M]
            # print('[DEBUG] pred_token_scores.shape:', pred_token_scores.shape)
            # pred_token_scores = pred_token_scores.reshape(-1, 1) # [BM, 1]
            # calculate the token score loss
            loss_token_scores = self.tokenscore_loss(pred_token_scores, indices, token_idx_manager)
            
            # print('[DEBUG] loss_token_scores:', loss_token_scores)
            
            loss_dict['loss_token_scores'] = loss_token_scores * self.loss_token_weight
        
        anchor_boxes_copy = box_cxcywh_to_xyxy(anchor_boxes_copy)
        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes_copy = anchor_boxes_copy[None].repeat(B, 1, 1)

        ious = []
        pos_ious = []
        for i in range(B):
            src_idx, tgt_idx = indices[i]
            # iou between predbox and tgt box
            iou, _ = box_iou(pred_box_copy[i, ...], (targets[i]['boxes']).clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                # In self.matcher, one gt box can match top-k predboxes, now we take the max iou gt box for each predbox
                max_iou = iou.max(dim=1)[0] 
            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes_copy[i], (targets[i]['boxes']).clone())
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)

        ious = torch.cat(ious)
        ignore_idx = ious > self.cfg['igt']
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.cfg['iou_t']

        src_idx = torch.cat(
            [src + idx * anchor_boxes_copy[0].shape[0] for idx, (src, _) in
             enumerate(indices)])
        # [BM,]
        gt_cls = torch.full(pred_cls.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=self.device)
        gt_cls[ignore_idx] = -1
        tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        tgt_cls_o[pos_ignore_idx] = -1

        gt_cls[src_idx] = tgt_cls_o.to(self.device)

        foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # cls loss
        masks = outputs['mask']
        valid_idxs = (gt_cls >= 0) & masks
        
        # print('[DEBUG] before loss_labels')
        
        # if self.weight_map is not None:
        #     h, w = self.weight_map.shape
        #     weights = self.weight_map.repeat(B, int(pred_cls.shape[0]/(B*h*w)), 1, 1)
        #     weights = weights.view(-1)
        #     weights = weights.to(self.device)

        #     loss_labels = self.loss_labels(pred_cls[valid_idxs], 
        #                                gt_cls_target[valid_idxs], 
        #                                num_foreground, weights[valid_idxs])
        # else:
        #     loss_labels = self.loss_labels(pred_cls[valid_idxs], 
        #                                 gt_cls_target[valid_idxs], 
        #                                 num_foreground)
        
        
        # Don't re-weight the cls loss
        loss_labels = self.loss_labels(pred_cls[valid_idxs], 
                                       gt_cls_target[valid_idxs], 
                                       num_foreground)

        loss_dict['cls_loss'] = loss_labels * self.loss_cls_weight
        
        # print('[DEBUG] after loss_labels')
        # print('[DEBUG] loss_labels:', loss_labels)

        # loss_labels = self.loss_labels(pred_cls[valid_idxs], 
        #                                 gt_cls_target[valid_idxs], 
        #                                 num_foreground)

        # box loss
        tgt_boxes = torch.cat([t['boxes'][i]
                                    for t, (_, i) in zip(targets, indices)], dim=0).to(self.device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = pred_box.reshape(-1, 4)[src_idx[~pos_ignore_idx]]

        # print('[DEBUG] before loss_bboxes')

        try:
            loss_bboxes = self.loss_bboxes(matched_pred_box, 
                                        tgt_boxes, 
                                        num_foreground)
        except Exception as e:
            print('Error in loss_bboxes')
            print(e)
            # print('matched_pred_box', matched_pred_box)
            # print('tgt_boxes', tgt_boxes)
            # print('num_foreground', num_foreground)
            raise e
        
        # print('[DEBUG] after loss_bboxes')
        # print('[DEBUG] loss_bboxes:', loss_bboxes)

        loss_dict['reg_loss'] = loss_bboxes * self.loss_reg_weight

        # total loss
        losses = torch.tensor(0.0, device=self.device)
        for k, v in loss_dict.items():
            losses += v
            
        loss_dict['total_loss'] = losses

        return loss_dict
    



def heatmap_ce(
    pred: torch.Tensor, 
    gt: torch.Tensor,
    # sigmoid: bool = True,
    reduction: str = 'mean'
    ):
    """
    This function will calculate the cross entropy loss between the predicted heatmap and the ground truth heatmap.
    
    Assume the input is a flatten tensor with shape (N, H*W, C), where N is the batch size, H is the height of the heatmap, W is the width of the heatmap, and C is the number of classes.
    
    For single sample, single class the cross entropy loss is calculated as:
    
    H^c(pred, gt) = \frac{-1}{\sum_{i,j} pred^c_{i,j}} \sum_{i,j} gt^c_{i,j} \log(pred^c_{i,j})
    
    Args:
    - pred: a flatten tensor with shape (N, H*W, C), the predicted heatmap
    - gt: a flatten tensor with shape (N, H*W, C), the ground truth heatmap
    
    Returns:
    - loss: the cross entropy loss between the predicted heatmap and the ground truth heatmap
    
    """
    
    # pred = pred.sigmoid() if sigmoid else pred
    pred = torch.nn.LogSoftmax(dim=1)(pred)
    
    gt = torch.nn.functional.softmax(gt, dim=1)
    
    loss = -gt * pred # (N, H*W, C)
    
    loss = loss.sum(dim=1) # (N, C)
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"reduction {reduction} is not supported")
    
    return loss

class Heatmap_CrossEntropy(nn.Module):
    
    def __init__(self, reduction='mean'):
        super(Heatmap_CrossEntropy, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred_hm, tgt_hm):
        """
            pred_hm: (Tensor) [B, H*W, C]
            tgt_hm:  (Tensor) [B, H*W, C]
            
            
        """
        loss_hm = heatmap_ce(pred_hm, tgt_hm, reduction=self.reduction)
        
        return loss_hm


# -------- helpers (internal) --------
def _gt_norm_over_space(gt_BSC, eps=1e-12):
    """Normalize nonnegative GT by sum over space per (B,C). Return probs and presence mask."""
    Z = gt_BSC.clamp_min(0).sum(dim=1, keepdim=True)              # (B,1,C)
    present = (Z.squeeze(1) > eps)                                # (B,C) boolean
    G = (gt_BSC / Z.clamp_min(eps)).clamp_min(eps)                # (B,S,C)
    return G, present

def _pred_spatial_probs_from_logits(pred_BSC, eps=1e-12):
    """Treat pred as logits over space per class; softmax over S."""
    # (B,S,C) -> (B,C,S) for stable softmax across space
    D_BCS = F.softmax(pred_BSC.permute(0,2,1), dim=-1)            # (B,C,S)
    D_BSC = D_BCS.permute(0,2,1).clamp_min(eps)                   # (B,S,C)
    return D_BSC

def _pred_spatial_probs_from_scores(pred_scores_BSC, eps=1e-12):
    """Treat pred as nonnegative scores; normalize by SUM over space per (B,C)."""
    Z = pred_scores_BSC.clamp_min(0).sum(dim=1, keepdim=True).clamp_min(eps)  # (B,1,C)
    return (pred_scores_BSC / Z).clamp_min(eps)                                # (B,S,C)

# -------- cross-entropy (paper-style) --------
def heatmap_2dce(
    pred: torch.Tensor,   # (B,S,C) logits OR scores OR GT (see input_type)
    gt: torch.Tensor,     # (B,S,C) nonnegative GT heatmaps
    reduction: str = 'mean',
    input_type: str = 'pred_logits'  # 'pred_logits'|'pred_probs'|'gt'
):
    """
    Normalized cross-entropy over space per class:
      Hx = -(1/log S) * sum_x G(x) * log D(x)

    GT handling:
      - G is always GT normalized by SUM over space (no softmax) and defines presence.
    Pred handling (via input_type):
      - 'pred_logits' : pred are spatial LOGITS per class -> D = softmax over space
      - 'pred_probs'  : pred are nonnegative spatial SCORES -> D = sum-normalize over space
      - 'gt'          : pred is another GT-like map (sum-normalized). Use this for CE(GT,GT) sanity check by passing pred==gt.

    Absent classes (sum over space == 0 in GT) are ignored in the reduction.
    """
    assert pred.shape == gt.shape and pred.dim() == 3, "Expected shapes (B, S, C)"
    B, S, C = pred.shape
    logS = torch.log(torch.tensor(S, dtype=pred.dtype, device=pred.device)).clamp_min(1e-12)

    # G & presence from GT
    G, present = _gt_norm_over_space(gt)               # (B,S,C), (B,C)

    # D from pred according to input_type
    if input_type == 'pred_logits':
        D = _pred_spatial_probs_from_logits(pred)
    elif input_type == 'pred_probs':
        D = _pred_spatial_probs_from_scores(pred)
    elif input_type == 'gt':
        # Treat pred as GT-like scores; sum-normalize over space.
        D, _ = _gt_norm_over_space(pred)
    else:
        raise ValueError("input_type must be 'pred_logits', 'pred_probs', or 'gt'.")

    # Cross-entropy over space per class (normalized by log S)
    CE_BC = -(G * D.log()).sum(dim=1) / logS           # (B,C)

    # mask out absent classes
    CE_BC = CE_BC.masked_fill(~present, float('nan'))

    if reduction == 'none':
        return CE_BC
    if reduction == 'sum':
        return torch.nan_to_num(CE_BC, nan=0.0).sum()
    # mean over present entries only
    return torch.nanmean(CE_BC)

class Heatmap_CrossEntropy_2D(nn.Module):
    def __init__(self, reduction='mean', input_type='pred_logits'):
        super().__init__()
        self.reduction = reduction
        self.input_type = input_type
    def forward(self, pred_hm, tgt_hm):
        """
        pred_hm: (B, H*W, C) (logits/scores/GT-like depending on input_type)
        tgt_hm:  (B, H*W, C) nonnegative GT heatmaps
        """
        return heatmap_2dce(pred_hm, tgt_hm, reduction=self.reduction, input_type=self.input_type)

# ---------- NEW: small helper for class selection ----------
def _select_classes(X_BSC: torch.Tensor, class_idx):
    """
    Select one or more classes on the last dimension.
    class_idx can be: int, list[int], or 1D LongTensor.
    Returns: X_sel (B,S,C_sel) and an index tensor idx (C_sel,)
    """
    C = X_BSC.size(-1)
    if isinstance(class_idx, int):
        idx = torch.tensor([class_idx], dtype=torch.long, device=X_BSC.device)
    elif isinstance(class_idx, (list, tuple)):
        idx = torch.tensor(class_idx, dtype=torch.long, device=X_BSC.device)
    elif isinstance(class_idx, torch.Tensor):
        idx = class_idx.to(dtype=torch.long, device=X_BSC.device)
    else:
        raise ValueError("class_idx must be int, list[int], or 1D LongTensor")

    if (idx.min() < 0) or (idx.max() >= C):
        raise IndexError(f"class_idx out of range [0,{C-1}]")
    return X_BSC.index_select(dim=-1, index=idx), idx

# ---------- Self-entropy with class selection ----------
def heatmap_self_entropy_2d(
    X: torch.Tensor,                     # (B,S,C)
    input_type: str = 'gt',              # 'gt' | 'pred_logits' | 'pred_probs'
    reduction: str = 'mean',
    ignore_absent_classes: bool = True,
    class_idx=None                       # NEW: int | list[int] | 1D LongTensor | None
):
    """
    Normalized self-entropy over space per class subset:
      H = -(1/log S) * sum_x P(x) * log P(x)

    If class_idx is provided, compute SE only for those classes (on C axis).
    - 'gt'         : X are GT scores; sum-normalize over space; absent classes can be ignored.
    - 'pred_logits': X are spatial logits; softmax over space per class; all counted.
    - 'pred_probs' : X are nonnegative scores; sum-normalize over space; all counted.
    """
    assert X.dim() == 3, "Expected (B, S, C)"
    B, S, C = X.shape
    logS = torch.log(torch.tensor(S, dtype=X.dtype, device=X.device)).clamp_min(1e-12)

    # Select classes if requested
    if class_idx is not None:
        X, idx = _select_classes(X, class_idx)   # (B,S,C_sel)
        C_sel = X.size(-1)
    else:
        C_sel = C

    # Build spatial distributions
    if input_type == 'gt':
        P, present = _gt_norm_over_space(X)                       # (B,S,C_sel), (B,C_sel)
        if not ignore_absent_classes:
            present = torch.ones(B, C_sel, dtype=torch.bool, device=X.device)
    elif input_type == 'pred_logits':
        P = _pred_spatial_probs_from_logits(X)                    # (B,S,C_sel)
        present = torch.ones(B, C_sel, dtype=torch.bool, device=X.device)
    elif input_type == 'pred_probs':
        P = _pred_spatial_probs_from_scores(X)                    # (B,S,C_sel)
        present = torch.ones(B, C_sel, dtype=torch.bool, device=X.device)
    else:
        raise ValueError("input_type must be 'gt', 'pred_logits', or 'pred_probs'.")

    # Self-entropy per (B, selected C)
    H_BC = -(P * P.log()).sum(dim=1) / logS                       # (B,C_sel)
    H_BC = H_BC.masked_fill(~present, float('nan'))

    if reduction == 'none':
        return H_BC                   # (B,C_sel)
    if reduction == 'sum':
        return torch.nan_to_num(H_BC, nan=0.0).sum()
    return torch.nanmean(H_BC)        # mean over present entries only

class Heatmap_SelfEntropy_2D(nn.Module):
    def __init__(self, input_type='gt', reduction='mean',
                 ignore_absent_classes=True, class_idx=None):
        super().__init__()
        self.input_type = input_type
        self.reduction = reduction
        self.ignore_absent_classes = ignore_absent_classes
        self.class_idx = class_idx    # NEW

    def forward(self, X):
        """
        X: (B, H*W, C)
        """
        return heatmap_self_entropy_2d(
            X,
            input_type=self.input_type,
            reduction=self.reduction,
            ignore_absent_classes=self.ignore_absent_classes,
            class_idx=self.class_idx
        )


def build_criterion(args, cfg, device, num_classes=80):
    
    if 'loss_type' in cfg:
        if cfg['loss_type'] == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif cfg['loss_type'] == 'hm_ce':
            criterion = Heatmap_CrossEntropy()
        elif cfg['loss_type'] == 'hm_ce_2d':
            criterion = Heatmap_CrossEntropy_2D() if 'loss_cfg' not in cfg else Heatmap_CrossEntropy_2D(**cfg['loss_cfg'])
        elif cfg['loss_type'] == 'hm_self_entropy_2d':
            criterion = Heatmap_SelfEntropy_2D(**cfg['loss_cfg']) if 'loss_cfg' in cfg else Heatmap_SelfEntropy_2D()
        elif cfg['loss_type'] == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError('Unknown loss type: {}'.format(cfg['loss_type']))
        return criterion

    if 're_weight_type' in cfg:
        assert args.train_min_size == args.train_max_size, 're-weight-type only support fixed size training'
        h, w = args.train_min_size // 16, args.train_max_size // 16
        
        if 're_weight_config' in cfg:
            weight_map = generate_weight_crop_uncrop_gt((3*h, 3*w), (h, w), **cfg['re_weight_config'])
        else:
            weight_map = generate_weight_crop_uncrop_gt((3*h, 3*w), (h, w))

        criterion = Criterion(cfg=cfg,
                            device=device,
                            loss_cls_weight=args.loss_cls_weight,
                            loss_reg_weight=args.loss_reg_weight,
                            loss_hm_weight=args.loss_hm_weight,
                            weight_map=weight_map,
                            num_classes=num_classes,
                            num_anchor=len(cfg['anchor_size']))
    else:
        criterion = Criterion(cfg=cfg,
                            device=device,
                            loss_cls_weight=args.loss_cls_weight,
                            loss_reg_weight=args.loss_reg_weight,
                            loss_hm_weight=args.loss_hm_weight,
                            num_classes=num_classes,
                            num_anchor=len(cfg['anchor_size']))
    return criterion

    
if __name__ == "__main__":
    pass
