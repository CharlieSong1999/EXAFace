# YOLOH config
import copy

from .yoloh_config import yoloh_config

template = 'yoloh_expand-50-DC5-640-expand-attn-2enc-2dec'

name = 'yoloh_expand-50-DC5-640-expand-attn-2enc-2dec-C2F_tconv_decoder_lq0-noposvalue-rope-residual-tconv-2scale-2'
yoloh_config[name] = copy.deepcopy(yoloh_config[template])
yoloh_config[name]['expansion_cfg']['attn']['tconv_decoder'] = True
yoloh_config[name]['expansion_cfg']['attn']['learnbale_query_size'] = 0
yoloh_config[name]['topk'] = 6
yoloh_config[name]['expansion_cfg']['attn']['no_pos_encoding_in_self_attn_values'] = True
yoloh_config[name]['expansion_cfg']['attn']['coarse_to_fine'] = True
yoloh_config[name]['expansion_cfg']['attn']['pos_type'] = 'rope'
yoloh_config[name]['expansion_cfg']['attn']['C2F_residual'] = True
yoloh_config[name]['expansion_cfg']['attn']['residual_type'] = 'tconv'
yoloh_config[name]['expansion_cfg']['attn']['granularities'] = [2, 1]
yoloh_config[name]['expansion_cfg']['attn']['reduce_factor'] = 2
