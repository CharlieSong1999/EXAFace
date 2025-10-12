"""
This code was originally obtained from:
# https://github.com/meta-llama/codellama/blob/main/llama/model.py
https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py
"""

import torch
import torch.nn as nn
import math
from functools import partial
import warnings
from einops import rearrange

def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compress_all_coords(end_x: int,
                        end_y: int,
                        patch_size: int = 4,
                        device=None,
                        dtype=torch.float32):
    """
    Compress the entire end_x x end_y grid into non-overlapping patch centers.

    Returns:
        t_x: (num_patches,) x-coordinates of patch centers
        t_y: (num_patches,) y-coordinates of patch centers

    Notes:
        - Order is raster: by patch rows (top to bottom), then patch cols (left to right).
        - Requires end_x and end_y are divisible by patch_size.
    """
    assert end_x % patch_size == 0 and end_y % patch_size == 0, \
        "end_x and end_y must be multiples of patch_size"

    # Top-left coords of each patch
    px = torch.arange(0, end_x, patch_size, device=device, dtype=dtype)  # along x
    py = torch.arange(0, end_y, patch_size, device=device, dtype=dtype)  # along y
    PX, PY = torch.meshgrid(px, py, indexing='xy')  # shape: (num_px, num_py)

    half = (patch_size - 1) * 0.5
    centers_x = (PX + half).reshape(-1).to(dtype)
    centers_y = (PY + half).reshape(-1).to(dtype)

    return centers_x, centers_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def compute_axial_cis_given_xy(dim: int, t_x: torch.Tensor, t_y: torch.Tensor, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-4], x.shape[-2], x.shape[-1]):
        # The freqs_cis is [B, HxW, C] and x is [B, num_heads, HxW, C]
        shape = [d if i != 1 else 1 for i, d in enumerate(x.shape)]

    try:
        return freqs_cis.view(*shape)
    except Exception as e:
        print(f"freqs_cis shape: {freqs_cis.shape}, x shape: {x.shape}")
        raise ValueError(f"Cannot reshape freqs_cis {freqs_cis.shape} to match x {x.shape}") from e

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis = freqs_cis.to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def apply_rotary_emb_single(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    freqs_cis = freqs_cis.to(x_.device)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x).to(x.device)


def average_downsample(x: torch.Tensor, factor: int, dim_first: bool = True):
    if not dim_first:
        if x.ndim == 4:
            x = rearrange(x, 'b h w c -> b c h w')
        elif x.ndim == 3:
            x = rearrange(x, 'h w c -> c h w')
    
    # Can't use nn.AvgPool2d because it doesn't support ComplexFloat
    _h, _w = x.shape[-2], x.shape[-1]
    assert _h % factor == 0 and _w % factor == 0, "Height and width must be divisible by the factor"
    nh, nw = _h // factor, _w // factor
    nx = x.new_zeros((*x.shape[:2], nh, nw) if x.ndim == 4 else (*x.shape[:1], nh, nw))
    if x.ndim == 4:
        for i in range(nh):
            for j in range(nw):
                # print(x[..., i*factor:(i+1)*factor, j*factor:(j+1)*factor].shape)
                # print()
                nx[..., i, j] = x[..., i*factor:(i+1)*factor, j*factor:(j+1)*factor].mean(dim=(-2, -1))
    elif x.ndim == 3:
        for i in range(nh):
            for j in range(nw):
                # print(x[..., i*factor:(i+1)*factor, j*factor:(j+1)*factor].shape)
                # print(x[..., i*factor:(i+1)*factor, j*factor:(j+1)*factor].mean(dim=(-2,-1)).shape)
                nx[..., i, j] = x[..., i*factor:(i+1)*factor, j*factor:(j+1)*factor].mean(dim=(-2, -1))
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.ndim}. Expected 3 or 4 dimensions, got {x.ndim}.")
    
    if not dim_first:
        if x.ndim == 4:
            nx = rearrange(nx, 'b c h w -> b h w c')
        elif x.ndim == 3:
            nx = rearrange(nx, 'b c h -> b h c')
            
    return nx

def rope_MultiheadAttention_forward(self, num_heads, q_mlp, k_mlp, v_mlp, Q, K, V, freqs_cis, need_weights=False):
    B, N, C = Q.shape
    q, k, v = q_mlp(Q), k_mlp(K), v_mlp(V)
    # Multi-head attention expects q, k, v to be of shape (B, N, num_heads, C // num_heads)
    q = q.reshape(B, N, num_heads, C // num_heads).permute(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, C // num_heads).permute(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, C // num_heads).permute(0, 2, 1, 3)

    # Apply rotary position embedding
    
    q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

    if need_weights:
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    else:
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop.p, is_causal=False
        )

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    if need_weights:
        return x, attn
    else:
        return x

    
class MultiheadAttention_with_pos_embed(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0, pos_type='add', **kwargs):
        super(MultiheadAttention_with_pos_embed, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.q_mlp = nn.Linear(dim, dim)
        self.k_mlp = nn.Linear(dim, dim)
        self.v_mlp = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_type = pos_type  # 'add' or 'rope' or 'rope_mixed, https://arxiv.org/abs/2403.13298
        print(f"Using pos_type: {self.pos_type}")
        print(kwargs)
        
        if self.pos_type not in ['add', 'rope', 'rope_mix', 'rope_and_ape']:
            raise ValueError(f"pos_type must be one of 'add', 'rope', 'rope_mix', or 'rope_and_ape' but got {self.pos_type}")
        
    def forward(self, q, k, v, mask=None, q_pos=None, k_pos=None, need_weights=False):
        B, N_q, C = q.shape
        _, N_k, _ = k.shape
        
        if self.pos_type == 'add':
            if q_pos is not None:
                q = q + q_pos
            if k_pos is not None:
                k = k + k_pos
        elif self.pos_type == 'rope_and_ape':
            assert q_pos is not None and k_pos is not None, "q_pos and k_pos must be provided for rope_and_ape"
            q = q + q_pos['ape']
            k = k + k_pos['ape']

        q = self.q_mlp(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_mlp(k).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_mlp(v).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.pos_type in ['rope', 'rope_mix']:
            # Apply rotary position embedding
            assert q_pos is not None and k_pos is not None, "q_pos and k_pos must be provided for rope or rope_mix"
            q, k = apply_rotary_emb_single(q, freqs_cis=q_pos), apply_rotary_emb_single(k, freqs_cis=k_pos)
        elif self.pos_type == 'rope_and_ape':
            # Apply rotary position embedding with additional absolute positional encoding
            assert q_pos is not None and k_pos is not None, "q_pos and k_pos must be provided for rope_and_ape"
            q = apply_rotary_emb_single(q, freqs_cis=q_pos['rope'])
            k = apply_rotary_emb_single(k, freqs_cis=k_pos['rope'])
        
        if need_weights:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
            
            attn = attn.mean(dim=1)  # Average attention weights across heads
            attn = attn.resize(B, N_q, N_k)  # Reshape to (B, N_q, N_k)
            
        else:
            # If need_weights is False, this can be more efficient
            x = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p, is_causal=False
            )
            x = x.transpose(1, 2).reshape(B, N_q, C)
            attn = None  # No attention weights returned
        
        return x, attn