import torch
from .yoloh import YOLOH
from .yoloh_expand import YOLOH_Expand

name2model = {
    'yoloh': YOLOH,
    'yoloh_expand': YOLOH_Expand,
}


# build YOLOH detector
def build_model(args, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=True,
                coco_pretrained=None,
                fig_size=(640, 640)):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if 'model' in cfg.keys():
        MODEL = name2model[cfg['model']]
    else:
        MODEL = YOLOH
    
    model = MODEL(cfg=cfg,
                  device=device, 
                  num_classes=num_classes, 
                  trainable=trainable,
                  conf_thresh=args.conf_thresh,
                  nms_thresh=args.nms_thresh,
                  topk=args.topk,
                  fig_size=fig_size)

    # Load COCO pretrained weight
    if coco_pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(coco_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)
                        
    return model


def build_model_noargs( 
                cfg, 
                device, 
                num_classes=80, 
                trainable=True,
                coco_pretrained=None,
                fig_size=(640, 640),
                **kwargs
                ):
    print('==============================')
    print('Build without args, with {} kwargs ...'.format(kwargs))
    
    
    if 'model' in cfg.keys():
        MODEL = name2model[cfg['model']]
    else:
        MODEL = YOLOH
    
    model = MODEL(cfg=cfg,
                  device=device, 
                  num_classes=num_classes, 
                  trainable=trainable,
                  conf_thresh=kwargs.get('conf_thresh', 0),
                  nms_thresh=kwargs.get('nms_thresh', 0),
                  topk=kwargs.get('topk', 10000),
                  fig_size=fig_size)

    # Load COCO pretrained weight
    if coco_pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(coco_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)
                        
    return model
