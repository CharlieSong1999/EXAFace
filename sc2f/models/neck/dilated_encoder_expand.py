import torch.nn as nn
from ..basic.conv import Conv
from utils import weight_init
import torch
from einops import rearrange


# Dilated Encoder
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 dilation=1,
                 expand_ratio=0.25,
                 act_type='relu'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        
        

    def forward(self, x):
        return x + self.branch(x)


class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 dilation=[4, 8, 12, 16],
                 expand_ratio=0.25,
                 act_type='relu'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch0 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[0], d=dilation[0], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch1 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[1], d=dilation[1], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[2], d=dilation[2], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        self.branch3 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation[3], d=dilation[3], act_type=act_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type)
        )
        # self.branch4 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[4], d=dilation[4], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch5 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[5], d=dilation[5], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch6 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[6], d=dilation[6], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )
        # self.branch7 = nn.Sequential(
        #     Conv(in_dim, inter_dim, k=1, act_type=act_type),
        #     Conv(inter_dim, inter_dim, k=3, p=dilation[7], d=dilation[7], act_type=act_type),
        #     Conv(inter_dim, in_dim, k=1, act_type=act_type)
        # )

    def forward(self, x):
        x1 = self.branch0(x) + x
        x2 = self.branch1(x1 + x) + x1
        x3 = self.branch2(x2 + x1 + x) + x2
        x4 = self.branch3(x3 + x2 + x1 + x) + x3
        return x4

class DilatedEncoder_expand(nn.Module):
    """ DilateEncoder """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio=0.25, 
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu', cfg=None, **kwargs):
        super(DilatedEncoder_expand, self).__init__()
        #  ***************************** conv
        self.projector2_1 = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None)
        )
        self.projector1_1 = nn.Sequential(
            Conv(out_dim*2, out_dim, k=1, act_type=None)
        )
        # self.projector0_1 = nn.Sequential(
        #     Conv(out_dim//2, out_dim, k=1, act_type=None)
        # )
        self.projector2 = nn.Sequential(
            Conv(out_dim, out_dim, k=3, p=1, act_type=None)
        )

        #  ***************************** avgpool
        self.avgpool_2x2 = nn.AvgPool2d((2, 2), stride=(2, 2))
        # self.avgpool_4x4 = nn.AvgPool2d((4, 4), stride=(4, 4))
        #  ***************************** AdaptiveAvgPool2d
        # self.ada_avgpool = nn.AdaptiveAvgPool2d((25, 42))

        # encoders = []
        # for d in dilation_list:
        #     encoders.append(Bottleneck(in_dim=out_dim,
        #                                dilation=d,
        #                                expand_ratio=expand_ratio,
        #                                act_type=act_type))

        encoders = []
        encoders.append(Bottleneck(in_dim=out_dim, dilation=dilation_list, expand_ratio=expand_ratio, act_type=act_type))
        self.encoders = nn.Sequential(*encoders)
        
        if cfg is not None:
            if 'pad_method_before_dilated' in cfg.keys():
                if cfg['pad_method_before_dilated'] == 'constant':
                    self.pad_method_before_dilated = 'constant'
                elif cfg['pad_method_before_dilated'] == 'learnable':
                    self.pad_method_before_dilated = 'learnable'
                    self.learnable_pad_token = nn.Parameter(torch.zeros(512))
                    torch.nn.init.normal_(self.learnable_pad_token, std=0.02)
                else:
                    raise ValueError('Unknown pad method: {}'.format(cfg['pad_method_before_dilated']))
            else:
                self.pad_method_before_dilated = 'constant'

        self._init_weight()

    def _init_weight(self):
        # for m in self.projector0_1:
        #     if isinstance(m, nn.Conv2d):
        #         weight_init.c2_xavier_fill(m)
        #         weight_init.c2_xavier_fill(m)
        #     if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        for m in self.projector1_1:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.projector2_1:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.projector2:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
        # x = self.projector(x)
        # x = self.encoders(x)
    # resnet50:      x2 2048*40*40   x1 1024*40*40   x0 512*80*80
    # cspdarknet53   x2 1024*40*40   x1 512*40*40   x0 256*80*80
    # To 512*n*n
    def forward(self, x):
        # print(x[0].size())
        # print(x[1].size())
        # print(x[2].size())
        x2 = self.projector2_1(x[2])
        x1 = self.projector1_1(x[1])
        # x0 = self.projector0_1(x[0])
        x0 = self.avgpool_2x2(x[0])
        # x0 = self.ada_avgpool(x0)
        # x1 = self.avgpool_2x2(x1)
        x = x2 + x1 + x0
        x = self.projector2(x)
        
        h, w = x.size(2), x.size(3)
        
        before_dilated = x.clone()
        
        x = nn.functional.pad(x, (h, h, w, w), mode='constant', value=0)
        if self.pad_method_before_dilated == 'learnable':
            x = rearrange(x, 'b c h w -> b h w c')
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool)
            mask[:, h:-h, w:-w] = 1
            x[~mask,:] = self.learnable_pad_token
            x = rearrange(x, 'b h w c -> b c h w')
            
        x = self.encoders(x)

        return x, before_dilated
