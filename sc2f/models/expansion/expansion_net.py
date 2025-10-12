import torch 
from torch import nn
from ..basic.conv import Conv
from .attn_expander import Feat_Expansion_Attn, Feat_Expansion_Attn_C2F



def get_Upsample_layer(layer_type='none',num_channel=256, scale_factor=3, fig_size=(640,640), *args, **kwargs):
    if layer_type == 'attn':
        if 'attn' not in kwargs.keys():
            raise ValueError('Upsample layer type is attn, but attn config is not provided')
        if 'coarse_to_fine' in kwargs['attn'].keys() and kwargs['attn']['coarse_to_fine']:
            return Feat_Expansion_Attn_C2F(
                scale_factor=scale_factor,
                fig_size=fig_size,
                **kwargs['attn']
            )
        else:
            return Feat_Expansion_Attn(
                scale_factor=scale_factor,
                fig_size=fig_size,
                **kwargs['attn']
            )
    elif layer_type == 'none':
        return nn.Identity()
    else:
        raise ValueError('Unknown Upsample layer type: {}'.format(layer_type))