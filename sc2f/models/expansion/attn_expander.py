import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import warnings
from utils import distributed_utils
from functools import partial

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from ..basic.conv import Conv
from .expansion_utils import TokenIndexManager
from .rope_attn import MultiheadAttention_with_pos_embed, compute_axial_cis, average_downsample, init_2d_freqs, init_t_xy, compute_mixed_cis, compute_axial_cis_given_xy, compress_all_coords 

import numpy as np
from torch import Tensor
import os

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")
    
Has_print_dict = {}


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        self.scaling = self.d_k ** -0.5 # Scaling factor for dot product
        # self.attn_dropout = nn.Dropout(attn_dropout) # Dropout layer
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pre_norm=False, **kwargs):
        super(EncoderLayer, self).__init__()
        pos_type = kwargs.get('pos_type', 'add')
        self.self_attn = MultiheadAttention_with_pos_embed(d_model, num_heads, pos_type=pos_type)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        
    def forward(self, x, mask, pos_embed=None, need_weights=False):
        
        if self.pre_norm:
            x = self.norm1(x)
        
        Q=K=V=x

        attn_output = self.self_attn(Q, K, V, mask, pos_embed, pos_embed, need_weights=need_weights)[0]
        x = self.norm1(x + self.dropout(attn_output)) if not self.pre_norm else x + self.dropout(attn_output)
        ff_output = self.feed_forward(x) if not self.pre_norm else self.feed_forward(self.norm2(x))
        x = self.norm2(x + self.dropout(ff_output)) if not self.pre_norm else x + self.dropout(ff_output)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pre_norm=False, no_pos_encoding_in_self_attn_values=False, *args, **kwargs):
        super(DecoderLayer, self).__init__()
        pos_type = kwargs.get('pos_type', 'add')
        self.self_attn = MultiheadAttention_with_pos_embed(d_model, num_heads, pos_type=pos_type)
        self.cross_attn = MultiheadAttention_with_pos_embed(d_model, num_heads, pos_type=pos_type)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.no_pos_encoding_in_self_attn_values = no_pos_encoding_in_self_attn_values
        
    def forward(self, x, enc_output, src_mask, tgt_mask, query_pos_embed=None, key_pos_embed=None, need_weights=False):
        """
        x: B, num_tokens_expanded, d_model
        enc_output: B, num_tokens_expanded, d_model
        src_mask: B, num_tokens_expanded
        tgt_mask: B, num_tokens_expanded
        query_pos_embed: B, num_tokens_expanded, d_model
        key_pos_embed: B, num_tokens_expanded, d_model
        
        """
        
        Q=x
        K=V=enc_output
        
        if self.pre_norm:
            Q = self.norm1(Q)
        
        if self.no_pos_encoding_in_self_attn_values:
            # If no positional encoding in self-attention values, use Q as V
            attn_output = self.self_attn(Q, Q, Q, tgt_mask, query_pos_embed, query_pos_embed, need_weights=need_weights)[0]
        else:
            assert self.self_attn.pos_type == 'add', "Only 'add' pos_type is supported for having pos embed in self-attn values"
            # Otherwise, use Q as both query and value
            attn_output = self.self_attn(Q, Q, Q + query_pos_embed, tgt_mask, query_pos_embed, query_pos_embed, need_weights=need_weights)[0]

        x = self.norm1(x + self.dropout(attn_output)) if not self.pre_norm else x + self.dropout(attn_output)
        
        if self.pre_norm:
            Q = self.norm2(Q)
        else:
            Q = x

        attn_output = self.cross_attn(Q, K, V, src_mask, query_pos_embed, key_pos_embed, need_weights=need_weights)[0]
        x = self.norm2(x + self.dropout(attn_output)) if not self.pre_norm else x + self.dropout(attn_output)
        ff_output = self.feed_forward(x) if not self.pre_norm else self.feed_forward(self.norm3(x))
        x = self.norm3(x + self.dropout(ff_output)) if not self.pre_norm else x + self.dropout(ff_output)
        return x
    

class Feat_Expansion_Attn(nn.Module):
    def __init__(self, d_model=512, fig_size=(640,640), num_heads=8, d_ff=512, num_dec_layers=1, 
                 num_enc_layers=0, dropout=0, scale_factor=3, reduce_factor=4, fest_proj=False,
                 pre_norm=False, mlp_final=False, *args, **kwargs):
        super(Feat_Expansion_Attn, self).__init__()
        self.d_model = d_model
        
        fig_h, fig_w = fig_size
        fest_h, fest_w = fig_h // 16, fig_w // 16
        scaled_h, scale_w = fest_h * scale_factor, fest_w * scale_factor
        reduce_h, reduce_W = scaled_h // reduce_factor, scale_w // reduce_factor
        num_tokens_expanded = reduce_h * reduce_W * 8//9
        
        assert num_tokens_expanded * 9 // 8 == reduce_h * reduce_W, 'num_tokens_expanded * 9 // 8 should be equal to reduce_h * reduce_W'
        
        print(f'num_tokens_expanded: {num_tokens_expanded}')
        
        self.num_tokens_expanded = num_tokens_expanded
        self.scale_factor = scale_factor
        self.reduce_factor = reduce_factor
        self.num_dec_layers = num_dec_layers
        self.num_enc_layers = num_enc_layers
        
        

        assert 2 ** int(np.log2(reduce_factor)) == reduce_factor, 'reduce_factor should be power of 2'

        if 'tconv_decoder' in kwargs:
            
            assert 2 ** int(np.log2(reduce_factor)) == reduce_factor, 'reduce_factor should be power of 2'
            
            if (type(kwargs['tconv_decoder']) == bool and kwargs['tconv_decoder']) or (type(kwargs['tconv_decoder']) == int and kwargs['tconv_decoder'] == -1):
                # default tconv_decoder
                self.tconv_decoder = nn.ModuleList([nn.Sequential(
                    nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2, padding=0),  # Upscale by 3
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(),
                ) for _ in range(int(np.log2(reduce_factor)))])
                self.tconv_decoder = nn.Sequential(*self.tconv_decoder)
            elif type(kwargs['tconv_decoder']) == int and kwargs['tconv_decoder'] == 1:
                self.tconv_decoder = nn.Sequential(
                    nn.ConvTranspose2d(d_model, d_model, kernel_size=self.reduce_factor, stride=self.reduce_factor, padding=0),  # Upscale by 3
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(),
                ) 
                
            else:
                print(f'tconv-decoder is not used')
            
            kwargs.pop('tconv_decoder')

        if 'out_conv' in kwargs:
            if type(kwargs['out_conv']) == bool and kwargs['out_conv']:
                print(f'out_conv: 2 layers')
                self.out_conv = nn.Sequential(
                    Conv(d_model, d_model, k=3, s=1, p=1),
                    Conv(d_model, d_model, k=3, s=1, p=1),
                )
            elif type(kwargs['out_conv']) == int:
                print(f'out_conv: {kwargs["out_conv"]} layers')
                self.out_conv = nn.Sequential(
                    **[Conv(d_model, d_model, k=3, s=1, p=1) for _ in range(kwargs['out_conv'])]
                )
            else:
                print(f'out_conv is not used')
            kwargs.pop('out_conv')

        self.expanded_reassign_coordinates = None
        
        if 'num_tokens_expanded' in kwargs:
            kwargs.pop('num_tokens_expanded')
        
        self.attn_fest_query = Attn_Fest_Query(d_model, num_tokens_expanded, num_heads, d_ff, num_dec_layers, 
                                               num_enc_layers, dropout, scale_factor, reduce_factor, fest_proj,
                                               pre_norm, *args, **kwargs)

    def forward(self, x, trainable=True, concatenate=True):
        B, C, H, W = x.shape
        
        expanded_H, expanded_W = H*self.scale_factor, W*self.scale_factor
        
        global Has_print_dict
        
        if distributed_utils.is_main_process():
            if trainable:
                radom_cos_input = random_mutual_cos_sim(x[0].detach().cpu().permute(1,2,0).view(-1, C).numpy(), x[0].detach().cpu().permute(1,2,0).view(-1, C).numpy())
                print(f'radom_cos_input: {radom_cos_input}')
        
        if 'before_attn_fest_query' not in Has_print_dict:
            print(f'x.dtype: {x.dtype}, x.shape: {x.shape}')
            Has_print_dict['before_attn_fest_query'] = True

        expanded_features = self.attn_fest_query(x) # B, num_tokens_expanded, d_model

        if self.expanded_reassign_coordinates is None:
            x_index = torch.arange(0, self.reduce_factor)
            y_index = torch.arange(0, self.reduce_factor)
            x_index, y_index = torch.meshgrid(x_index, y_index)
            coordinates = torch.stack((x_index, y_index), dim=2)
            expanded_reassign_coordinates = []
            
            assert expanded_H % self.reduce_factor == 0, f'expanded_H({expanded_H}) mod reduce_factor({self.reduce_factor}) should be 0'
            
            for i in range(expanded_H//self.reduce_factor):
                for j in range(expanded_W//self.reduce_factor):
                    if (expanded_H//3*2//self.reduce_factor) > i >= (expanded_H//3//self.reduce_factor) and (expanded_W//3*2//self.reduce_factor) > j >= (expanded_W//3//self.reduce_factor):
                        continue
                    expanded_reassign_coordinates.append(coordinates + torch.tensor([i * self.reduce_factor, j * self.reduce_factor]))
            expanded_reassign_coordinates = torch.stack(expanded_reassign_coordinates, dim=0)
            self.expanded_reassign_coordinates = expanded_reassign_coordinates # on cpu by default

        expanded_reassign_coordinates_on_device = self.expanded_reassign_coordinates.to(x.device)

        
        expanded_reassign_coordinates_on_device = rearrange(expanded_reassign_coordinates_on_device, 'n h w c -> (n h w) c')

        expanded_features = rearrange(expanded_features, 'b n f -> b n f 1 1 ')
        expanded_features_array = []
        
        if 'before_tconv_decoder' not in Has_print_dict:
            print(f'Before tconv-decoder --> expanded_features.dtype: {expanded_features.dtype}, expanded_features.shape: {expanded_features.shape}')
            Has_print_dict['before_tconv_decoder'] = True
        
        for i in range(expanded_features.shape[0]):
            # print(expanded_features[i].shape)
            if not hasattr(self, 'tconv_decoder'):
                expanded_features_array.append(nn.functional.interpolate(expanded_features[i], scale_factor=self.reduce_factor, mode='bilinear', align_corners=False))
            else:
                expanded_features_array.append(self.tconv_decoder(expanded_features[i]))
                
        expanded_features = torch.stack(expanded_features_array, dim=0) # B, num_tokens_expanded, d_model, reduce_factor, reduce_factor
        expanded_features = rearrange(expanded_features, 'b n d h w -> b (n h w) d')
        
        if 'after_tconv_decoder' not in Has_print_dict:
            print(f'After tconv-decoder --> expanded_features.dtype: {expanded_features.dtype}, expanded_features.shape: {expanded_features.shape}')
            Has_print_dict['after_tconv_decoder'] = True
        
        if not concatenate:
            # if don't concatenate, return the expanded features and the coordinates
            # expanded_features: B, num_tokens_expanded, d_model
            # expanded_reassign_coordinates_on_device: num_tokens_expanded, 2
            return expanded_features, expanded_reassign_coordinates_on_device
        
        if torch.isnan(expanded_features).any():
            raise ValueError('expanded_features has nan')

        full_features = torch.nn.functional.pad(x, (H, H, W, W), value=0)
        full_features = rearrange(full_features, 'b c h w -> b h w c')
        
        if full_features.dtype == torch.float16 and expanded_features.dtype == torch.float32:
            expanded_features = expanded_features.half()
            

        full_features[:, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_features
        
        # full_features = self.attn_fest_query.fest_proj_out(full_features)
        
        full_features = rearrange(full_features, 'b h w c -> b c h w')
        
        if hasattr(self, 'out_conv'):
            full_features = self.out_conv(full_features)

        return full_features

class Attn_Fest_Query_C2F(nn.Module):
    def __init__(self,  d_model=512, num_tokens_expanded=800, num_heads=8, d_ff=512, num_dec_layers=1, 
                 num_enc_layers=0, dropout=0, scale_factor=3, reduce_factor=4, fest_proj=False,
                 pre_norm=False, *args, **kwargs):
        super(Attn_Fest_Query_C2F, self).__init__()
        self.kwargs = kwargs
        if num_enc_layers > 0:
            self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, pre_norm, **kwargs) for _ in range(num_enc_layers)])
        
        self.fest_proj = fest_proj
        self.d_model = d_model
        self.num_heads = num_heads
        self.need_attn_weights = False # Default to be False, can be set to True if needed. Note that this will increase memory usage.
        
        self.granularities = kwargs.get('granularities', [4,2,1])  # 'fine' or 'coarse'
        
        self.tconv_decoder = nn.ModuleList([nn.Sequential(
                    nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2, padding=0),  # Upscale by 3
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(),
                ) for _ in range(int(np.log2(reduce_factor)))])
        self.tconv_decoder = nn.Sequential(*self.tconv_decoder)

        decoderlayer = DecoderLayer
            
        self.downsample_type = 'repeat'
            
        if 'C2F_residual' in kwargs:
            if kwargs['C2F_residual']:
                print(f'C2F_residual: True')
                self.C2F_residual = True
                # self.queries_per_gid = {}
            else:
                print(f'C2F_residual: False')
                self.C2F_residual = False
            kwargs.pop('C2F_residual')
            
            if 'residual_type' in kwargs:
                self.residual_type = kwargs['residual_type']
            else:
                self.residual_type = 'query'
        else:
            print(f'C2F_residual: False')
            self.C2F_residual = False
            self.residual_type = 'query'
            
        def _make_layers(layer_class, _num_layers, *args, **kwargs):
            return nn.ModuleList([layer_class(*args, **kwargs) for _ in range(_num_layers)])
        
        def _make_layers_dict(layer_class, _num_layers, keys, *args, **kwargs):
            # Note that nn.ModuleDict requires keys to be strings, so we convert int keys to strings
            # However, in other part of the code, we might use int keys.
            keys = [str(k) for k in keys if isinstance(k, int)]
            return nn.ModuleDict({key: _make_layers(layer_class, _num_layers, *args, **kwargs) for key in keys}) if len(keys) > 0 else _make_layers(layer_class, _num_layers, *args, **kwargs)

        def _make_layer(layer_class, *args, **kwargs):
            return layer_class(*args, **kwargs)
        
        def _make_layer_dict(layer_class, keys, *args, **kwargs):
            keys = [str(k) for k in keys if isinstance(k, int)]
            return nn.ModuleDict({key: _make_layer(layer_class, *args, **kwargs) for key in keys}) if len(keys) > 0 else _make_layer(layer_class, *args, **kwargs)


            # print(f'decoder_per_granularity: False')
        self.scoring = _make_layer(
            DetrMLPPredictionHead,
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=1,
            num_layers=2,
        )
            # self.decoder_per_granularity = False


        self.decoder = _make_layers_dict(
            decoderlayer, num_dec_layers, 
            [], 
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, pre_norm=pre_norm, **kwargs)
        
            
        self.scale_factor = scale_factor
        self.reduce_factor = reduce_factor
        self.num_tokens_expanded = num_tokens_expanded
        
        self.pos_embed = DetrSinePositionEmbedding(d_model//2, normalize=True)
        self.pos_type = kwargs.get('pos_type', 'add')

        if self.pos_type not in ['add', 'rope', 'rope_mix', 'rope_and_ape']:
            raise ValueError(f'Unknown pos_type: {self.pos_type}, should be one of "add", "rope", "rope_mix", "rope_and_ape"')

        if self.pos_type == 'rope_mix':
            self.compute_cis = partial(compute_axial_cis, num_heads=num_heads)
            
            freqs = init_2d_freqs(
                dim=d_model // num_heads, num_heads=num_heads,theta=10, rotate=True,
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        elif self.pos_type in ['rope', 'rope_and_ape']:
            self.compute_cis = partial(compute_axial_cis, dim=d_model // num_heads, theta=10)
        

        self.coordinates_for_expanded = None
        self.expanded_reassign_coordinates = None
        self.pos_for_expanded = None
        self.other_kwargs = kwargs
        
        if 'learnbale_query_size' in kwargs:
            if kwargs['learnbale_query_size'] == 1:
                self.token_queries = nn.Embedding(1, d_model)
            elif kwargs['learnbale_query_size'] == -1:
                self.token_queries = nn.Embedding(num_tokens_expanded, d_model)
            elif kwargs['learnbale_query_size'] == 0:
                pass
            else:
                raise ValueError(f'learnbale_query_size should be 0 or 1, but got {kwargs["learnbale_query_size"]}')
        else:
            self.token_queries = nn.Embedding(num_tokens_expanded, d_model)
        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        self.apply(_init_weights)
        
        if 'learnbale_query_zero_init' in kwargs and kwargs['learnbale_query_zero_init']:
            nn.init.constant_(self.token_queries.weight, 0)


    def _decode_tokens(self, layers, queries, flatten_x, pos_embed, rearranged_pos_x, **kwargs):
        for lid, layer in enumerate(layers):
            queries = layer(queries, flatten_x, None, None, pos_embed, rearranged_pos_x, need_weights=self.need_attn_weights)
        
        return queries # B, num_tokens_expanded, d_model
    
    def _tconv_decode(self, queries, decoder,**kwargs):
        """Decode the queries with tconv decoder.
        
        Args:
            queries (torch.Tensor): The queries to be decoded, shape (B, num_tokens, d_model).
            decoder (nn.Module): The tconv decoder to be used.
            
        Returns:
            torch.Tensor: The decoded queries, shape (B, num_tokens_expanded (often x4), d_model).
        """
        B, num_tokens, d_model = queries.shape
        
        queries = rearrange(queries, 'b n c -> (b n) c 1 1')
        queries = decoder(queries)  # (B * num_tokens_expanded, d_model, 1, 1)
        queries = rearrange(queries, '(b n) c h w -> b (n h w) c', b=B, n=queries.shape[0] // B)

        return queries
    
    def _score(self, queries, scoring, g_id, queries_per_gid=None):
        """Score the queries with the scoring head.
        
        Args:
            queries (torch.Tensor): The queries to be scored, shape (B, num_tokens_expanded, d_model).
            granularity (int): The granularity level for scoring.

        Returns:
            torch.Tensor: The scores for the queries, shape (B, num_tokens_expanded).
        """
        granularity = str(self.granularities[g_id])
        queries = queries.clone()  # Ensure queries are not modified in place
        
        
        if self.C2F_residual:
            for _gid in range(g_id):
                assert queries_per_gid is not None, 'queries_per_gid should not be None when C2F_residual is True'
                prev_queries = queries_per_gid[_gid].clone()  # Clone to avoid modifying the original queries
                prev_queries = prev_queries.repeat_interleave(
                    4, dim=1
                )
                queries += prev_queries # B, num_tokens_expanded, d_model
        
        score = scoring(queries) # B, num_tokens_expanded, 1
            
        score = rearrange(score, 'b n 1 -> b n')  # B, num_tokens_expanded
        return score
    
    def _get_score_func(self, granularity):
        """Get the scoring function for the given granularity.
        
        Args:
            granularity (int): The granularity level for scoring.
            
        Returns:
            nn.Module: The scoring function for the given granularity.
        """
        
        return self.scoring
        
    def _get_decoder_func(self, granularity):
        """Get the decoder function for the given granularity.
        
        Args:
            granularity (int): The granularity level for decoding.
            
        Returns:
            nn.Module: The decoder function for the given granularity.
        """
        
        return self.decoder

    def forward(self, x):

        B, C, H, W = x.shape


        if self.fest_proj:
            x = self.fest_proj_in(x)

        Expanded_H, Expanded_W = H*self.scale_factor, W*self.scale_factor

        if not hasattr(self, '_pos_embed'):
            self._pos_embed = self.pos_embed(x, torch.ones((B, Expanded_H, Expanded_W), device=x.device)) # B, d_model, 3H, 3W
            Expanded_pos = self._pos_embed
        else:
            if self._pos_embed.shape != (B, self.d_model, Expanded_H, Expanded_W):
                # update pos_embed
                self._pos_embed = self.pos_embed(x, torch.ones((B, Expanded_H, Expanded_W), device=x.device))
                Expanded_pos = self._pos_embed
            else:
                # use cached pos_embed
                Expanded_pos = self._pos_embed

        if not hasattr(self, '_rope_pos_embed') and (self.pos_type in ['rope', 'rope_mix', 'rope_and_ape']):
            # use rope pos_embed
            
            if self.pos_type == 'rope_mix':
                if not hasattr(self, 'freqs_t_x'):
                    t_x, t_y = init_t_xy(end_x=Expanded_H, end_y=Expanded_W)
                    t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                    self.freqs_t_x = t_x
                    self.freqs_t_y = t_y
                else:
                    t_x, t_y = self.freqs_t_x, self.freqs_t_y
                    
                assert hasattr(self, 'freqs'), 'freqs should be initialized before using rope pos_embed'
                freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            elif self.pos_type in ['rope', 'rope_and_ape']:
                freqs_cis = self.compute_cis(end_x=Expanded_H, end_y=Expanded_W)
                
            
            freqs_cis = rearrange(freqs_cis, '(h w) d -> 1 d h w', h=Expanded_H, w=Expanded_W).repeat(B, 1, 1, 1) # B, d_model, H, W
            freqs_cis = freqs_cis.to(x.device)

            self._rope_pos_embed = freqs_cis
        elif hasattr(self, '_rope_pos_embed'):
            # update rope pos_embed
            if self.pos_type == 'rope_mix':
                freqs_cis = self.compute_cis(self.freqs, self.freqs_t_x, self.freqs_t_y)
                freqs_cis = rearrange(freqs_cis, '(h w) d -> 1 d h w', h=Expanded_H, w=Expanded_W).repeat(B, 1, 1, 1) # B, d_model, H, W
            elif self.pos_type in ['rope', 'rope_and_ape'] and self._rope_pos_embed.shape != (B, self.d_model, Expanded_H, Expanded_W):
                freqs_cis = self.compute_cis(end_x=Expanded_H, end_y=Expanded_W)
                freqs_cis = rearrange(freqs_cis, '(h w) d -> 1 d h w', h=Expanded_H, w=Expanded_W).repeat(B, 1, 1, 1)
            else:
                freqs_cis = self._rope_pos_embed
            freqs_cis = freqs_cis.to(x.device)
            self._rope_pos_embed = freqs_cis

        # print('Expanded_pos.shape', Expanded_pos.shape)

        pos_x = Expanded_pos[:, :, H:2*H, W:2*W] # B, d_model, H, W
        
        if 'learnbale_query_size' not in self.other_kwargs or self.other_kwargs['learnbale_query_size'] == -1:
            # default use num_tokens_expanded as query size
            queries = self.token_queries.weight.unsqueeze(0).repeat(B, 1, 1) # B, num_tokens_expanded, d_model
        elif self.other_kwargs['learnbale_query_size'] == 0:
            # if learnbale_query_size is 0, use zeros as queries
            queries = torch.zeros(B, self.num_tokens_expanded, self.d_model, device=x.device)
        elif self.other_kwargs['learnbale_query_size'] == 1:
            # if learnbale_query_size is 1, use single learnable query
            queries = self.token_queries.weight.unsqueeze(0).repeat(B, self.num_tokens_expanded, 1)
        else:
            raise ValueError(f'learnbale_query_size should be 0 or 1, but got {self.other_kwargs["learnbale_query_size"]}')

        # print(queries.shape, pos_for_expanded.shape)

        flatten_x = rearrange(x, 'b c h w -> b (h w) c')
        
        if self.pos_type == 'add':
            rearranged_pos_x = rearrange(pos_x, 'b d h w -> b (h w) d')
        elif self.pos_type in ['rope', 'rope_and_ape', 'rope_mix']:
            # use rope pos_embed
            rearranged_pos_x = rearrange(self._rope_pos_embed[:, :, H:2*H, W:2*W], 'b d h w -> b (h w) d')
            if self.pos_type == 'rope_and_ape':
                # add absolute positional encoding
                rearranged_pos_x_ape = rearrange(pos_x, 'b d h w -> b (h w) d')
                rearranged_pos_x = {
                    'ape': rearranged_pos_x_ape,
                    'rope': rearranged_pos_x
                }
        else:
            raise ValueError(f'Unknown pos_type: {self.pos_type}, should be one of "add", "rope", "rope_mix", "rope_and_ape"')

        if hasattr(self, 'encoder'):
            for lid, layer in enumerate(self.encoder):
                flatten_x = layer(flatten_x, None, rearranged_pos_x, need_weights=self.need_attn_weights)
                if torch.isnan(flatten_x).any():
                    raise ValueError(f'flatten_x from {lid}encoder has nan')
        
        all_scores = []
        all_queries = []
        token_idx_manager = TokenIndexManager(B, H, W, self.scale_factor, x.device)
        token_idx_manager._init_pos_to_idx_table_per_gradularity(self.granularities)
        
        granularity = self.granularities[0]
        token_idx_manager.allocate_coarse_token_ids(granularity)
        pos_coarse_token = token_idx_manager.token_pos_per_granularity[granularity]
        
        
        if not hasattr(self, 'pos_embed_per_granularity'):
            self.pos_embed_per_granularity = {}
            self.pos_embed_per_granularity[self.granularities[-1]] = Expanded_pos.clone() 
            for granularity in self.granularities[:-1]:
                self.pos_embed_per_granularity[granularity] = torch.nn.AvgPool2d(
                    kernel_size=(granularity, granularity), stride=(granularity, granularity)
                )(self.pos_embed_per_granularity[self.granularities[-1]])
                
            if self.pos_type in ['rope', 'rope_mix', 'rope_and_ape']:
                self.rope_pose_embed_per_granularity = {}
                self.rope_pose_embed_per_granularity[self.granularities[-1]] = self._rope_pos_embed.clone()
                for granularity in self.granularities[:-1]:
                    rope_compress_type = 'avg_pool' if not 'rope_compress_type' in self.kwargs else self.kwargs['rope_compress_type']
                    if rope_compress_type == 'avg_pool':
                        self.rope_pose_embed_per_granularity[granularity] = average_downsample(self.rope_pose_embed_per_granularity[self.granularities[-1]], granularity)
                    elif rope_compress_type == 'center_sample':
                        compressed_compute_cis = partial(compute_axial_cis_given_xy, dim=self.d_model // self.num_heads, theta=10)
                        compressed_t_x, compressed_t_y = compress_all_coords(Expanded_W, Expanded_H, patch_size=granularity)
                        freqs_cis_compressed = compressed_compute_cis(t_x=compressed_t_x, t_y=compressed_t_y)
                        freqs_cis_compressed = freqs_cis_compressed.to(self._rope_pos_embed.device)
                        self.rope_pose_embed_per_granularity[granularity] = rearrange(freqs_cis_compressed, '(h w) d -> 1 d h w', h=Expanded_H//granularity, w=Expanded_W//granularity).repeat(B, 1, 1, 1)
        else:
            if self.pos_embed_per_granularity[self.granularities[-1]].shape != Expanded_pos.shape:
                self.pos_embed_per_granularity[self.granularities[-1]] = Expanded_pos.clone() 
                for granularity in self.granularities[:-1]:
                    self.pos_embed_per_granularity[granularity] = torch.nn.AvgPool2d(
                        kernel_size=(granularity, granularity), stride=(granularity, granularity)
                    )(self.pos_embed_per_granularity[self.granularities[-1]])
                    
            if self.pos_type in ['rope', 'rope_mix', 'rope_and_ape']:
                if self.rope_pose_embed_per_granularity[self.granularities[-1]].shape != self._rope_pos_embed.shape:
                    self.rope_pose_embed_per_granularity[self.granularities[-1]] = self._rope_pos_embed.clone()
                    for granularity in self.granularities[:-1]:
                        rope_compress_type = 'avg_pool' if not 'rope_compress_type' in self.kwargs else self.kwargs['rope_compress_type']
                        if rope_compress_type == 'avg_pool':
                            self.rope_pose_embed_per_granularity[granularity] = average_downsample(self.rope_pose_embed_per_granularity[self.granularities[-1]], granularity)
                        elif rope_compress_type == 'center_sample':
                            compressed_compute_cis = partial(compute_axial_cis_given_xy, dim=self.d_model // self.num_heads, theta=10)
                            compressed_t_x, compressed_t_y = compress_all_coords(Expanded_W, Expanded_H, patch_size=granularity)
                            freqs_cis_compressed = compressed_compute_cis(t_x=compressed_t_x, t_y=compressed_t_y)
                            freqs_cis_compressed = freqs_cis_compressed.to(self._rope_pos_embed.device)
                            self.rope_pose_embed_per_granularity[granularity] = rearrange(freqs_cis_compressed, '(h w) d -> 1 d h w', h=Expanded_H//granularity, w=Expanded_W//granularity).repeat(B, 1, 1, 1)

        if self.pos_type == 'add':
            pos_embed = self.pos_embed_per_granularity[granularity] # B, d_model, H, W
            pos_embed = token_idx_manager.get_value_with_2d_index(pos_embed, pos_coarse_token, granularity) # B, num_tokens_expanded, d_model
            if self.other_kwargs['learnbale_query_size'] == 0:
                queries = pos_embed
        elif self.pos_type in ['rope', 'rope_mix', 'rope_and_ape']:
            if self.other_kwargs['learnbale_query_size'] == 0:
                q_pos_embed = self.pos_embed_per_granularity[granularity] # B, d_model, H, W
                try:
                    q_pos_embed = token_idx_manager.get_value_with_2d_index(q_pos_embed, pos_coarse_token, granularity) # B, num_tokens_expanded, d_model
                except Exception as e:
                    print(f'q_pos_embed.shape: {q_pos_embed.shape}, pos_coarse_token.shape: {pos_coarse_token.shape}, granularity: {granularity}')
                    print(f"Error occurred while getting q_pos_embed: {e}")
                queries = q_pos_embed
            pos_embed = self.rope_pose_embed_per_granularity[granularity]
            pos_embed = token_idx_manager.get_value_with_2d_index(pos_embed, pos_coarse_token, granularity) # B, num_tokens_expanded, d_model
            
            if self.pos_type == 'rope_and_ape':
                pos_embed_ape = self.pos_embed_per_granularity[granularity]
                pos_embed_ape = token_idx_manager.get_value_with_2d_index(pos_embed_ape, pos_coarse_token, granularity) # B, num_tokens_expanded, d_model
                pos_embed = {
                    'rope': pos_embed,
                    'ape': pos_embed_ape
                }
                
        else:
            raise ValueError(f'Unknown pos_type: {self.pos_type}, should be one of "add", "rope", "rope_mix"')

        # queries for the coarse granularity
        queries = self._decode_tokens(
            self._get_decoder_func(granularity), 
            queries, flatten_x, pos_embed, rearranged_pos_x
        ) # B, num_tokens_expanded, d_model
        
        if len(self.granularities) == 1:
            all_queries = self._tconv_decode(queries, self.tconv_decoder) # B, num_tokens_expanded, d_model
            all_scores = all_queries.clone()[..., 0] # B, num_tokens_expanded
            token_idx_manager.refine_token_indices(torch.tensor([[]*B], device=x.device), torch.arange(queries.shape[1], device=x.device).repeat(B, 1), granularity, 1)
            token_assignment = token_idx_manager.token_assignment
        else:
            token_scores = self._score(queries, self._get_score_func(granularity), 0) # B, num_tokens_expanded
            all_scores.append(token_scores)
            
            # Get the token ids with descending scores
            token_ids = torch.argsort(token_scores, dim=-1, descending=True) # B, num_tokens_expanded
            
            # top k% tokens to be refined, and the rest to be kept and use tconv to expand
            if self.kwargs.get('top_k_percent_4x4', None) is not None:
                num_tokens_to_refine = int(token_scores.shape[1] * self.kwargs['top_k_percent_4x4'])
            else:
                # Default to use 25% of tokens to refine
                num_tokens_to_refine = token_scores.shape[1] // 4
            token_idx_to_refine, _ = torch.sort(token_ids[:, :num_tokens_to_refine], dim=-1, descending=False) # B, num_tokens_to_refine
            token_idx_to_keep, _ = torch.sort(token_ids[:, num_tokens_to_refine:], dim=-1, descending=False) # B, num_tokens_expanded - num_tokens_to_refine

            # print(f'[DEBUG] queries.shape: {queries.shape}')
            # print(f'[DEBUG] token_idx_to_refine.shape: {token_idx_to_refine.shape}, token_idx_to_keep.shape: {token_idx_to_keep.shape}')
            # Use tconv decoder to expand the tokens that are not refined
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1) # B, 1
            tokens_to_keep = self._tconv_decode(queries[batch_idx, token_idx_to_keep], self.tconv_decoder) # B, num_tokens_expanded - num_tokens_to_refine, d_model
            all_queries.append(tokens_to_keep)

            if self.C2F_residual:
                queries_per_gid = {}
                queries_per_gid[0] = queries[batch_idx, token_idx_to_refine].clone() # B, num_tokens_to_refine, d_model
            else:
                queries_per_gid = None

            for g_id in range(1, len(self.granularities)):
                granularity = self.granularities[g_id]
                from_granularity = self.granularities[g_id - 1]

                # Allocate token ids for the current granularity
                token_idx_manager.refine_token_indices(
                    token_idx_to_refine,
                    token_idx_to_keep,
                    from_granularity,
                    granularity,
                )
                pos_token = token_idx_manager.token_pos_per_granularity[granularity]
                pos_embed = self.pos_embed_per_granularity[granularity] # B, d_model, H, W
                pos_embed = token_idx_manager.get_value_with_2d_index(pos_embed, pos_token, granularity) # B, num_tokens_expanded, d_model
                    
                if self.downsample_type == 'repeat':
                    queries = queries[batch_idx, token_idx_to_refine].repeat_interleave(
                        4, dim=1
                    )
                elif self.downsample_type == 'repeat_with_pos':
                    queries = queries[batch_idx, token_idx_to_refine].repeat_interleave(
                        4, dim=1
                    )
                    queries += pos_embed
                elif self.downsample_type == 'pos_only':
                    queries = pos_embed
                else:
                    raise ValueError(f'Unknown downsample_type in Attn_Feat_Query_C2F.forward: {self.downsample_type}')
                
                if self.pos_type in ['rope_mix', 'rope']:
                    pos_embed = self.rope_pose_embed_per_granularity[granularity]
                    pos_embed = token_idx_manager.get_value_with_2d_index(pos_embed, pos_token, granularity) # B, num_tokens_expanded, d_model
                elif self.pos_type == 'rope_and_ape':
                    pos_embed_rope = self.rope_pose_embed_per_granularity[granularity]
                    pos_embed_rope = token_idx_manager.get_value_with_2d_index(pos_embed_rope, pos_token, granularity)
                    pos_embed = {
                        'rope': pos_embed_rope,
                        'ape': pos_embed
                    }

                queries = self._decode_tokens(
                    self._get_decoder_func(granularity),
                    queries, flatten_x, pos_embed, rearranged_pos_x
                ) # B, num_tokens_expanded, d_model

                # If not the last granularity, we need to get the token scores for the current granularity
                token_scores = self._score(queries, self._get_score_func(granularity), g_id, queries_per_gid) # B, num_tokens_expanded
                all_scores.append(token_scores)
                
                if g_id != len(self.granularities) - 1:
                    # If not the last granularity, we select the top 25% tokens to refine and the rest to keep and use tconv to expand
                    # Get the token ids with descending scores
                    token_ids = torch.argsort(token_scores, dim=-1, descending=True) # B , num_tokens_expanded
                    # top k% tokens to be refined, and the rest to be kept and use tconv to expand
                    if self.kwargs.get('top_k_percent_2x2', None) is not None:
                        num_tokens_to_refine = int(token_scores.shape[1] * self.kwargs['top_k_percent_2x2'])
                    else:
                        # Default to use 25% of tokens to refine
                        num_tokens_to_refine = token_scores.shape[1] // 4
                    token_idx_to_refine, _ = torch.sort(token_ids[:, :num_tokens_to_refine], dim=-1, descending=False) # B, num_tokens_to_refine
                    token_idx_to_keep, _ = torch.sort(token_ids[:, num_tokens_to_refine:], dim=-1, descending=False) # B, num_tokens_expanded - num_tokens_to_refine
                    
                    # Use tconv decoder to expand the tokens that are not refined
                    tokens_to_keep = queries[batch_idx, token_idx_to_keep] # B, num_tokens_expanded - num_tokens_to_refine, d_model
                    if self.C2F_residual and self.residual_type == 'query':
                        for _gid in range(g_id):
                            queries_per_gid[_gid] = queries_per_gid[_gid].repeat_interleave(
                                4, dim=1
                            )
                            tokens_to_keep += queries_per_gid[_gid][batch_idx, token_idx_to_keep] # B, num_tokens_expanded - num_tokens_to_refine, d_model
                            queries_per_gid[_gid] = queries_per_gid[_gid][batch_idx, token_idx_to_refine].clone() # B, num_tokens_to_refine, d_model

                        queries_per_gid[g_id] = queries[batch_idx, token_idx_to_refine].clone() # B, num_tokens_to_refine, d_model
                        
                    tokens_to_keep = self._tconv_decode(tokens_to_keep, self.tconv_decoder[g_id:]) # B, (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                    
                    if self.C2F_residual and self.residual_type == 'tconv':
                        for _gid in range(g_id):
                            queries_per_gid[_gid] = self._tconv_decode(queries_per_gid[_gid], self.tconv_decoder[g_id - 1])
                            prev_queries = queries_per_gid[_gid][batch_idx, token_idx_to_keep] # B, (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                            prev_queries = self._tconv_decode(prev_queries, self.tconv_decoder[g_id:])
                            tokens_to_keep += prev_queries # B, (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                            queries_per_gid[_gid] = queries_per_gid[_gid][batch_idx, token_idx_to_refine].clone() # B, num_tokens_to_refine, d_model
                        queries_per_gid[g_id] = queries[batch_idx, token_idx_to_refine].clone() # B, num_tokens_to_refine, d_model
                    
                    all_queries.append(tokens_to_keep)

            if granularity != 1:
                # Now it is 2x2
                token_idx_manager.refine_token_indices(
                    torch.tensor([[]*B], device=x.device), torch.arange(queries.shape[1], device=x.device).repeat(B, 1), granularity, 1)
                queries = self._tconv_decode(queries, self.tconv_decoder[g_id:]) # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                if self.C2F_residual and self.residual_type == 'tconv':
                    for _gid in range(g_id):
                        prev_queries = queries_per_gid[_gid]
                        prev_queries = self._tconv_decode(prev_queries, self.tconv_decoder[g_id - 1:])
                        queries += prev_queries

                token_assignment = token_idx_manager.token_assignment
            else:

                if self.C2F_residual:
                    if self.residual_type == 'query':
                        for _gid in range(len(self.granularities)-1):
                            queries_per_gid[_gid] = queries_per_gid[_gid].repeat_interleave(
                                4, dim=1
                            )
                            queries += queries_per_gid[_gid] # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                    elif self.residual_type == 'tconv':
                        for _gid in range(len(self.granularities)-1):
                            prev_queries = queries_per_gid[_gid]
                            prev_queries = self._tconv_decode(prev_queries, self.tconv_decoder[-1])
                            queries += prev_queries # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
                    else:
                        raise ValueError(f'Unknown residual_type in Attn_Feat_Query_C2F.forward: {self.residual_type}')
                    
                token_assignment = torch.cat((token_idx_manager.token_assignment, token_idx_manager.token_pos_per_granularity[1]), dim=1)

            all_queries.append(queries) # B, num_tokens_expanded, d_model
            all_queries = torch.cat(all_queries, dim=1) # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
            all_scores = torch.cat(all_scores, dim=1) # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor
        
        if self.fest_proj:
            all_queries = self.fest_proj_out(all_queries)

        self.token_idx_manager = token_idx_manager

        return {
            'queries': all_queries,  # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
            'scores': all_scores,  # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor
            'token_assignment': token_assignment,  # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, 2
            'token_idx_manager': token_idx_manager,  # TokenIndexManager object
        }
    
class Feat_Expansion_Attn_C2F(nn.Module):
    def __init__(self, d_model=512, fig_size=(640,640), num_heads=8, d_ff=512, num_dec_layers=1, 
                 num_enc_layers=0, dropout=0, scale_factor=3, reduce_factor=4, fest_proj=False,
                 pre_norm=False, mlp_final=False, *args, **kwargs):
        super(Feat_Expansion_Attn_C2F, self).__init__()
        self.d_model = d_model
        self.kwargs = kwargs
        
        fig_h, fig_w = fig_size
        fest_h, fest_w = fig_h // 16, fig_w // 16
        scaled_h, scale_w = fest_h * scale_factor, fest_w * scale_factor
        reduce_h, reduce_W = scaled_h // reduce_factor, scale_w // reduce_factor
        num_tokens_expanded = reduce_h * reduce_W * 8//9
        
        assert num_tokens_expanded * 9 // 8 == reduce_h * reduce_W, 'num_tokens_expanded * 9 // 8 should be equal to reduce_h * reduce_W'
        
        
        self.num_tokens_expanded = num_tokens_expanded
        self.scale_factor = scale_factor
        self.reduce_factor = reduce_factor
        self.num_dec_layers = num_dec_layers
        self.num_enc_layers = num_enc_layers

        assert 2 ** int(np.log2(reduce_factor)) == reduce_factor, 'reduce_factor should be power of 2'

        self.expanded_reassign_coordinates = None

        self.attn_fest_query = Attn_Fest_Query_C2F(d_model, num_tokens_expanded, num_heads, d_ff, num_dec_layers,
                                                   num_enc_layers, dropout, scale_factor, reduce_factor, fest_proj,
                                                   pre_norm, *args, **kwargs)

    def forward(self, x, trainable=True, concatenate=True):
        B, C, H, W = x.shape
        
        expanded_H, expanded_W = H*self.scale_factor, W*self.scale_factor
        
        global Has_print_dict

        query_output = self.attn_fest_query(x) # B, num_tokens_expanded, d_model
        
        # if the attn_fest_query returns a dict, we need to extract the queries
        expanded_features = query_output['queries'] # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, d_model
        all_scores = query_output['scores'] # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor
        token_assignment = query_output['token_assignment'] # B, num_tokens_expanded + (num_tokens_expanded - num_tokens_to_refine) * reduce_factor * reduce_factor, 2
        token_idx_manager = query_output['token_idx_manager'] # TokenIndexManager object
            
        expanded_reassign_coordinates_on_device = token_assignment

        if not concatenate:
            # if don't concatenate, return the expanded features and the coordinates
            # expanded_features: B, num_tokens_expanded, d_model
            # expanded_reassign_coordinates_on_device: B, num_tokens_expanded, 2
            return {
                'expanded_features': expanded_features,
                'expanded_reassign_coordinates_on_device': expanded_reassign_coordinates_on_device,
                'token_scores': all_scores,
                'token_idx_manager': token_idx_manager,
            }

        full_features = torch.nn.functional.pad(x, (H, H, W, W), value=0)
        full_features = rearrange(full_features, 'b c h w -> b h w c')
        
        if full_features.dtype == torch.float16 and expanded_features.dtype == torch.float32:
            expanded_features = expanded_features.half()

        batch_idx = torch.arange(0, full_features.shape[0], device=full_features.device).unsqueeze(1)
        full_features[batch_idx, expanded_reassign_coordinates_on_device[:, 0], expanded_reassign_coordinates_on_device[:, 1], :] = expanded_features

        
        full_features = rearrange(full_features, 'b h w c -> b c h w')
        
        if hasattr(self, 'out_conv'):
            full_features = self.out_conv(full_features)

        return {
            'expanded_features': full_features,
            'expanded_reassign_coordinates_on_device': expanded_reassign_coordinates_on_device,
            'token_scores': all_scores,
            'token_idx_manager': token_idx_manager,
        }
    
# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x