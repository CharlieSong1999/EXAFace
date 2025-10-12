import torch
import numpy as np

def process_weight_map(W, keep_ratio=0.9, min_val=0.33, max_val=1.0, linear_rescale=True):
    """
    Filters and rescales a distance-like weight map:
    - Discards values below threshold (set to 0)
    - Linearly maps values in [threshold, max_val] → [0, 1]
    """
    assert W.dim() == 2, "Expecting 2D tensor"
    total_range = max_val - min_val
    threshold = max_val - (total_range * keep_ratio)

    # Create mask
    mask = W >= threshold

    # Clamp below threshold to threshold (so that they rescale to 0)
    W_clipped = torch.where(mask, W, torch.tensor(threshold, device=W.device))

    # Rescale
    W_rescaled = (W_clipped - threshold) / (max_val - threshold)
    W_rescaled = W_rescaled * mask.float()  # ensure below-threshold stays at 0
    
    if not linear_rescale:
        W_rescaled[W_rescaled > 0.0] = 1.0

    return W_rescaled

def generate_weight_crop_uncrop_gt(full_size, crop_size, *args, **kwargs):
    """
    Argements:
        full_size: (f_h: int, f_w: int) the size of full image
        crop_size: (c_h: int, c_w: int) the size of crop image

    Returns:
        weight: f_h x f_w tensor, the weight of each pixel in full image
    """

    f_h, f_w = full_size

    c_h, c_w = crop_size

    res_h, res_w = (f_h - c_h)//2, (f_w - c_w)//2

    # generate weight
    weight = torch.ones(full_size)

    # half diagonal
    d_max = np.sqrt(f_h ** 2 + f_w ** 2) / 2 # half diagonal

    # for each pixel in full image
    d_x = torch.arange(1, f_w+1).view(1, -1).repeat(f_h, 1)
    d_y = torch.arange(1, f_h+1).view(-1, 1).repeat(1, f_w)

    # distance to nearest edge
    d_x[d_x <= res_w] = res_w - d_x[d_x <= res_w] + 1
    d_x[d_x > f_w - res_w] = d_x[d_x > f_w - res_w] - f_w + res_w
    d_x[torch.logical_and(d_x > res_w, d_x <= f_w-res_w)] = 0

    d_y[d_y <= res_h] = res_h - d_y[d_y <= res_h] + 1
    d_y[d_y > f_h - res_h] = d_y[d_y > f_h - res_h] - f_h + res_h
    d_y[torch.logical_and(d_y > res_h, d_y <= f_h-res_h)] = 0

    d = torch.sqrt(d_x ** 2 + d_y ** 2) / d_max

    weight = 1 - d
    
    if 'weight_keep_ratio' in kwargs.keys():
        if 'weight_linear_rescale' in kwargs.keys():
            weight = process_weight_map(weight, keep_ratio=kwargs['weight_keep_ratio'], linear_rescale=kwargs['weight_linear_rescale'])
        else:
            weight = process_weight_map(weight, keep_ratio=kwargs['weight_keep_ratio'], linear_rescale=True)
    
    return weight


