from torch import optim


def build_optimizer(args, model, cfg):
    
    base_lr = args.base_lr
    backbone_lr = args.backbone_lr
    name = cfg['optimizer']
    momentum = cfg['momentum']
    weight_decay = cfg['weight_decay']
    freeze_yoloh = args.freeze_yoloh
    if 'optim_cfg' in cfg.keys():
        optim_cfg = cfg['optim_cfg']
    else:
        optim_cfg = None
    
    print('==============================')
    print('Optimizer: {}'.format(name))
    print('--momentum: {}'.format(momentum))
    print('--weight_decay: {}'.format(weight_decay))
    print('--freezed_yoloh: {}'.format(freeze_yoloh))
    print('--base_lr: {}'.format(base_lr))
    print('--backbone_lr: {}'.format(backbone_lr))
    if optim_cfg is not None:
        print('--optim_cfg: {}'.format(optim_cfg))
    print('--alter_train: {}'.format(args.alter_train))

    if args.alter_train > 0:
        expansion_param = [{'params': model.expansion.parameters()}]
        yoloh_param = [{'params': [p for n, p in model.named_parameters() if 'expansion' not in n and 'backbone' not in n],},
                       {'params': model.backbone.parameters(), 'lr': backbone_lr}]
        
        assert optim_cfg is not None, 'optim_cfg is required for alter_train'
        
        optimizer_expansion = optim.AdamW(expansion_param, **optim_cfg)
        optimizer_yoloh = optim.SGD(yoloh_param,lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        
        return optimizer_expansion, optimizer_yoloh

    if freeze_yoloh:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "expansion" in n and p.requires_grad]},
            {
                'params': [p for n, p in model.named_parameters() if "head" in n and p.requires_grad],
                'lr': backbone_lr,
            }
        ]
        for n, p in model.named_parameters():
            if "expansion" not in n and p.requires_grad and "head" not in n:
                p.requires_grad = False
    else:
        if model.expansion is None:
            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': base_lr},
                {"params": [p for n, p in model.backbone.named_parameters()], 'lr': backbone_lr}
            ]
        else:
            param_dicts = [
                {"params": [p for n, p in model.expansion.named_parameters()]},
                {"params": [p for n, p in model.named_parameters() if 'expansion' not in n and 'backbone' not in n], 'lr': base_lr},
                {"params": [p for n, p in model.backbone.named_parameters()], 'lr': backbone_lr}
            ]

    if name == 'sgd':
        optimizer = optim.SGD(param_dicts, 
                                lr=base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    elif name == 'adam':
        optimizer = optim.Adam(param_dicts, 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    elif name == 'adamw':
        if optim_cfg is not None:
            optimizer = optim.AdamW(param_dicts, 
                                    lr=base_lr,
                                    **optim_cfg)
        else:
            optimizer = optim.AdamW(param_dicts, 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    return optimizer
