# YOLOH config


yoloh_config = {
    'yoloh18': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'yoloh50': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'yoloh50-DC5': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'yoloh101': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101',
        'norm_type': 'FrozeBN',
        'stride': 32,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'yoloh101-DC5': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'yoloh50-DC5-640': {
        # input
        'EAD': False,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        'val_transform': [
                {'name': 'ToTensor'},
                {'name': 'Resize'},
                {'name': 'Normalize'},
        ],
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 4, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 6,
        'iou_t': 0.1,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 12000,  
        # ori 1500  best 12000
        'warmup_factor': 0.000083333,
        # ori 0.00066667  best 0.000083333
        'epoch': {
            '1x': {'max_epoch': 14,
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24,
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 512, 544, 576, 608, 640]},
            '3x': {'max_epoch': 36,
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 512, 544, 576, 608, 640]},
        },
    },
    
    'yoloh_expand-50-DC5-640-expand-attn-2enc-2dec': {
        # input
        'EAD': True,
        'format': 'RGB',
        'selected_val_images': [16,217,92,95, 292, 929, 4401, 7524, 5350, 5148],
        'model': 'yoloh_expand',
        'expansion_cfg': {
            'layer_type': 'attn',
            'num_channel': 512,
            'scale_factor': 3,
            'attn': {
                'd_model': 512,
                'num_heads': 16,
                'd_ff': 4096,
                'num_dec_layers': 2,
                'num_enc_layers': 2,
                'dropout': 0.2,
            }
        },
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            'ft':[
                  {'name': 'ToTensor'},
                  {'name': 'EAD_Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],
            '1x':[
                  {'name': 'ToTensor'},
                  {'name': 'EAD_Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[
                  {'name': 'ToTensor'},
                  {'name': 'EAD_Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[
                        {'name': 'ToTensor'},
                        {'name': 'EAD_Resize'},
                        {'name': 'Normalize'},
                        {'name': 'PadImage'}]},
        'val_transform': [
                {'name': 'ToTensor'},
                {'name': 'EAD_Resize'},
                {'name': 'Normalize'},
        ],
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 4, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 6,
        'iou_t': 0.1,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-2,
        'optim_cfg': {
                'betas': (0.9, 0.98),
                'eps': 1e-8,
        },
        'warmup': 'linear',
        'wp_iter': 2000,  
        # ori 1500  best 12000
        'warmup_factor': 0.000083333,
        # ori 0.00066667  best 0.000083333
        'epoch': {
            'ft': {'max_epoch': 8,
                    'lr_epoch': [6, 8], 
                    'multi_scale': None},
            '1x': {'max_epoch': 14,
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24,
                    'lr_epoch': [16, 22], 
                    'multi_scale': [480, 512, 544, 576, 608, 640]},
            '3x': {'max_epoch': 36,
                    'lr_epoch': [24, 33], 
                    'multi_scale': [480, 512, 544, 576, 608, 640]},
        },
    },

    'yoloh53-640': {
        # input
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x': [{'name': 'RandomHorizontalFlip'},
                   {'name': 'RandomShift', 'max_shift': 32},
                   {'name': 'ToTensor'},
                   {'name': 'Resize'},
                   {'name': 'Normalize'},
                   {'name': 'PadImage'}],

            '2x': [{'name': 'RandomHorizontalFlip'},
                   {'name': 'RandomShift', 'max_shift': 32},
                   {'name': 'ToTensor'},
                   {'name': 'Resize'},
                   {'name': 'Normalize'},
                   {'name': 'PadImage'}],

            '3x': [{'name': 'DistortTransform',
                    'hue': 0.1,
                    'saturation': 1.5,
                    'exposure': 1.5},
                   {'name': 'RandomHorizontalFlip'},
                   {'name': 'RandomShift', 'max_shift': 32},
                   {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                   {'name': 'ToTensor'},
                   {'name': 'Resize'},
                   {'name': 'Normalize'},
                   {'name': 'PadImage'}]},
        # model
        'backbone': 'cspdarknet53',
        'norm_type': 'BN',
        'stride': 16,
        'act_type': 'relu',
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [1, 2, 3, 4, 5, 6, 7, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 12000,
        'warmup_factor': 0.000083333,
        'epoch': {
            '1x': {'max_epoch': 12,
                   'lr_epoch': [8, 11],
                   'multi_scale': None},
            '2x': {'max_epoch': 24,
                   'lr_epoch': [16, 22],
                   'multi_scale': [480, 512, 544, 576, 608, 640]},
            '3x': {'max_epoch': 36,
                   'lr_epoch': [24, 33],
                   'multi_scale': [480, 512, 544, 576, 608, 640]},
        },
    },



}