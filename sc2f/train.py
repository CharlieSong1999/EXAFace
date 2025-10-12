from __future__ import division

import os
import argparse
import time
import random
import numpy as np
from matplotlib import pyplot as plt

import torchvision.transforms.functional as transF
from PIL import Image
from einops import rearrange

from torch.distributed.elastic.multiprocessing.errors import record

from time import gmtime, strftime

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset, SubsetRandomSampler

# from data.voc import VOCDetection
from data.coco import COCODataset
from data.coco_fb import COCODataset_FB
from data.coco_fb_diff import COCODataset_FB_Diff
from data.coco_fb_diff_ws import COCODataset_FB_Diff_withSource
from config.yoloh_config_expand import yoloh_config
from data.transforms import TrainTransforms, ValTransforms, BaseTransforms

from utils import distributed_utils
from utils.criterion import build_criterion
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup, build_scheduler
from utils.solver.warmup_schedule import build_transformer_warmup

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from logger.wandbLogger import WandbAttentionLogger

from models.yoloh import build_model
import copy

# from transformers import get_linear_schedule_with_warmup

import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
os.environ['NCCL_DEBUG'] = "INFO"
os.environ['TORCH_CPP_LOG_LEVEL'] = "INFO"


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--schedule', type=str, default='1x', choices=['ft', '1x', '2x', '3x', '9x'],
                        help='training schedule. Attention, 9x is designed for YOLOH53-DC5.')
    parser.add_argument('-lr', '--base_lr', type=float, default=0.03,
                        help='base learning rate')
    parser.add_argument('-lr_bk', '--backbone_lr', type=float, default=0.01,
                        help='backbone learning rate')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')

    # input image size               
    parser.add_argument('--train_min_size', type=int, default=800,
                        help='The shorter train size of the input image')
    parser.add_argument('--train_max_size', type=int, default=1333,
                        help='The longer train size of the input image')
    parser.add_argument('--val_min_size', type=int, default=800,
                        help='The shorter val size of the input image')
    parser.add_argument('--val_max_size', type=int, default=1333,
                        help='The longer val size of the input image')

    # model
    parser.add_argument('-v', '--version', default='yoloh50',
                        help='build yoloh')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')
    parser.add_argument('--freeze_yoloh', action='store_true', default=False,)

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('--train_img_folder', default=None,
                        help='train_img_folder')
    parser.add_argument('--val_img_folder', default=None,
                        help='val_img_folder')
    parser.add_argument('--vis_img_folder', default=None,
                        help='vis_img_folder')
    parser.add_argument('--train_ann_file', default=None,)
    parser.add_argument('--val_ann_file', default=None,)
    parser.add_argument('--vis_ann_file', default=None,)
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # Loss
    parser.add_argument('--alpha', default=0.25, type=float,
                        help='focal loss alpha')
    parser.add_argument('--gamma', default=2.0, type=float,
                        help='focal loss gamma')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--loss_hm_weight', default=1.0, type=float,)
    
    parser.add_argument('--loss_cos_weight', default=1.0, type=float,)
    parser.add_argument('--loss_mse_weight', default=0.0, type=float, help='weight of mse loss, decrepted')
    
    # train trick
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='Mosaic augmentation')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('--half_precision', action='store_true', default=False,)
    parser.add_argument('--alter_train', type=int, default=0,)
    parser.add_argument('--reduce_steps', type=int, default=-1,)
    
    
    # dense guidence
    parser.add_argument('--dense_guidance', action='store_true', default=False,)
    parser.add_argument('--source_img_folder', default=None, type=str,)
    
    parser.add_argument('--model_lr', type=float, default=1.0,
                        help='base learning rate')
    parser.add_argument('--bk_lr', type=float, default=0.1,
                        help='backbone learning rate')
    parser.add_argument('--yoloh_lr', type=float, default=0.5,
                        help='base learning rate')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')
    
    parser.add_argument('--wandb_token', default=None, type=str,
                        help='wandb token')
    parser.add_argument('--exp_name', default='YOLOH', type=str,)

    parser.add_argument('--subset_ratio', default=1.0, type=float,)
    
    parser.add_argument('--skip_attention_map', action='store_true', default=False,)
    
    parser.add_argument('--manual_max_epoch', type=int, default=None)

    return parser.parse_args()

@record
def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print(f'torch version: {torch.__version__}')
    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda:{}".format(args.rank)) if args.distributed else torch.device("cuda")
    else:
        device = torch.device("cpu")

    # YOLOH Config
    if args.version not in yoloh_config.keys():
        raise ValueError(f"Invalid version {args.version}. Options are {yoloh_config.keys()}")

    cfg = yoloh_config[args.version]
    print('==============================')
    print('Model Configuration: \n', cfg)

    args.exp_name = f'{args.exp_name}-{strftime("%a%d%b%Y%H%M%S", gmtime())}'
    
    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    if distributed_utils.is_main_process():
        
        if args.wandb_token is None:
            print('No wandb token provided, wandb logging disabled')
            run = None
        else:
            wandb.login(key=args.wandb_token)
            run = wandb.init(
                # Set the project where this run will be logged
                project="Extreme-Amodal-detection",
                group=args.exp_name,
                name = f'{args.exp_name}',
                # Track hyperparameters and run metadata
                config={
                    "args": vars(args),
                    "cfg": cfg,
                },
            )
            
            # wandb.run.log_code('./',include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
            wandb.run.log_code(
                './',
                include_fn=lambda path: path.endswith(".py"),
            )
        
        if args.vis_img_folder is not None:
            # print('[DEBUG] Using vis_img_folder: ', args.vis_img_folder)
            # print('[DEBUG] Using vis_ann_file: ', args.vis_ann_file)
            vis_dataset = COCODataset_FB_Diff(
                            data_dir=args.vis_img_folder,
                            image_folder=args.vis_img_folder,
                            transform=None,
                            ann_file=args.vis_ann_file,)

            if run is not None:
                wandb_logger = WandbAttentionLogger(
                    run = run,
                    dataset=vis_dataset,
                    transform=evaluator.transform,
                    selected_indices= cfg['selected_val_images'] if 'selected_val_images' in cfg.keys() else list(range(0, 41, 4)),
                    cfg=cfg,
                    skip_attention_map=args.skip_attention_map,
                )
            else:
                wandb_logger = None
        else:
            wandb_logger = None

    if args.subset_ratio < 1.0:
        subset_indices = np.random.choice(len(dataset), int(len(dataset)*args.subset_ratio), replace=False)
        dataset = Subset(dataset, subset_indices)
        print('Subset of dataset created with ratio: ', args.subset_ratio)

    print('num_classes:', num_classes)

    # dataloader
    dataloader = build_dataloader(args, dataset, CollateFunc())

    # criterion
    criterion = build_criterion(args=args, device=device, cfg=cfg, num_classes=num_classes)
    
    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True,
                      coco_pretrained=args.coco_pretrained,
                      fig_size=(args.train_max_size, args.train_max_size))
    model = net
    model = model.to(device).train()

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_without_ddp.trainable = False
        model_without_ddp.eval()
        FLOPs_and_Params(model=model_without_ddp, 
                         min_size=args.train_min_size, 
                         max_size=args.train_max_size, 
                         device=device)
        model_without_ddp.trainable = True
        model_without_ddp.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()
        

    # optimizer
    optimizer = build_optimizer(args, model=model_without_ddp,
                                cfg = cfg,)
    
    
    if 'warmup_steps' in cfg.keys():
        if args.manual_max_epoch is not None:
            lr_scheduler = build_scheduler(args, optimizer, cfg, total_steps=len(dataloader)*args.manual_max_epoch, dim_embed=cfg['head_dim'],
                                       warmup_steps=cfg['warmup_steps'])
        else:
            lr_scheduler = build_scheduler(args, optimizer, cfg, total_steps=len(dataloader)*cfg['epoch'][args.schedule]['max_epoch'], dim_embed=cfg['head_dim'],
                                   warmup_steps=cfg['warmup_steps'])
    else:
        if args.manual_max_epoch is not None:
            lr_scheduler = build_scheduler(args, optimizer, cfg, total_steps=len(dataloader)*args.manual_max_epoch, dim_embed=cfg['head_dim'])
        else:
            lr_scheduler = build_scheduler(args, optimizer, cfg, total_steps=len(dataloader)*cfg['epoch'][args.schedule]['max_epoch'], dim_embed=cfg['head_dim'])
        

    # training configuration
    max_epoch = cfg['epoch'][args.schedule]['max_epoch']
    epoch_size = len(dataset) // (args.batch_size * args.num_gpu)
    best_map = -1.

    
    if args.half_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    t0 = time.time()
    # start training loop
    for epoch in range(max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, data in enumerate(dataloader):
            images, targets, masks = data
                
            ni = iter_i + epoch * epoch_size

            # to device
            images = images.to(device)
            masks = masks.to(device)

            if args.half_precision:
                # inference
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model_without_ddp(images, mask=masks)
                    
                    # compute loss
                    loss_output = criterion(outputs = outputs,
                                            targets = targets,
                                            anchor_boxes = model_without_ddp.anchor_boxes)
                    if isinstance(loss_output, tuple):
                        cls_loss, reg_loss, total_loss = loss_output
                        
                        loss_dict = dict(
                            cls_loss=cls_loss,
                            reg_loss=reg_loss,
                            total_loss=total_loss
                        )
                        
                    elif isinstance(loss_output, dict):
                        cls_loss = loss_output['cls_loss']
                        reg_loss = loss_output['reg_loss']
                        total_loss = loss_output['total_loss']
                        
                        loss_dict = loss_output
                    else:
                        raise ValueError('Invalid loss output type: {}'.format(type(loss_output)))
                    
                    loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

                
                scaler.scale(total_loss).backward()
                
                if args.grad_clip_norm > 0.:
                    total_norm = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), args.grad_clip_norm)
                else:
                    total_norm = get_total_grad_norm(model_without_ddp.parameters())
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            else:
                # inference
                outputs = model_without_ddp(images, mask=masks)

                # compute loss
                loss_output = criterion(outputs = outputs,
                                        targets = targets,
                                        anchor_boxes = model_without_ddp.anchor_boxes)
                if isinstance(loss_output, tuple):
                    cls_loss, reg_loss, total_loss = loss_output
                    
                    loss_dict = dict(
                        cls_loss=cls_loss,
                        reg_loss=reg_loss,
                        total_loss=total_loss
                    )
                elif isinstance(loss_output, dict):
                    loss_dict = loss_output
                else:
                    raise ValueError('Invalid loss output type: {}'.format(type(loss_output)))
                
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

                # Backward and Optimize
                total_loss.backward()
                if args.grad_clip_norm > 0.:
                    total_norm = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), args.grad_clip_norm)
                else:
                    total_norm = get_total_grad_norm(model_without_ddp.parameters())
                optimizer.step()
                optimizer.zero_grad()
        

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': -1, 'lr_yoloh': -1}
                
                if len(cur_lr) == 2:
                    cur_lr_dict['lr_bk'] = cur_lr[1]
                elif len(cur_lr) == 3:
                    cur_lr_dict['lr_bk'] = cur_lr[2]
                    cur_lr_dict['lr_yoloh'] = cur_lr[1]

                
                print('%s -> [Epoch %d/%d][Iter %d/%d][lr: %.6f][lr-yoloh: %.6f][lr-bk: %.6f][Loss: cls %.2f || reg %.2f || cos %.2f || gnorm: %.2f || size [%d, %d] || time: %.2f]'
                % (f'{strftime("%a%d%b%Y-%H%M%S", gmtime())}',
                    epoch+1, 
                    max_epoch, 
                    iter_i, 
                    epoch_size, 
                    cur_lr_dict['lr'],
                    cur_lr_dict['lr_yoloh'],
                    cur_lr_dict['lr_bk'],
                    loss_dict_reduced['cls_loss'].item() if 'cls_loss' in loss_dict_reduced.keys() else -1,
                    loss_dict_reduced['reg_loss'].item() if 'reg_loss' in loss_dict_reduced.keys() else -1,
                    loss_dict_reduced['cosim_loss'].item() if 'cosim_loss' in loss_dict_reduced.keys() else -1,
                    total_norm,
                    args.train_min_size, args.train_max_size, 
                    t1-t0),
                flush=True)
                    
                    
                wandb_dict = {
                    "epoch": epoch+1,
                    "iter": iter_i+1,
                    "total_loss": loss_dict_reduced['total_loss'].item(),
                    "total_grad_norm": total_norm,
                    "lr": cur_lr_dict['lr'],
                    # "size": [args.train_min_size, args.train_max_size],
                    "time": t1-t0
                }
                if len(cur_lr) == 2:
                    wandb_dict['lr_bk'] = cur_lr_dict['lr_bk']
                elif len(cur_lr) == 3:
                    wandb_dict['lr_bk'] = cur_lr_dict['lr_bk']
                    wandb_dict['lr_yoloh'] = cur_lr_dict['lr_yoloh']
                
                
                if run is not None:
                    wandb.log(wandb_dict)
                
                wandb_logger_log_iters = cfg['wandb_logger_log_iters'] if 'wandb_logger_log_iters' in cfg.keys() else 350
                assert wandb_logger_log_iters % 10 == 0, "wandb_logger_log_iters should be a multiple of 10"

                if (iter_i) % wandb_logger_log_iters == 0:
                    model_eval = model_without_ddp
                    model_eval.trainable = False
                    model_eval.eval()
                    
                    if wandb_logger is not None:
                        wandb_logger.log(model_eval, step=iter_i + epoch * epoch_size)
                    
                    # set train mode.
                    model_eval.trainable = True
                    model_eval.train()

                t0 = time.time()

            lr_scheduler.step()
        
        # evaluation
        
        if distributed_utils.is_main_process():
            print('Saving state, epoch:', epoch + 1)
            weight_name = '{}_epoch_{}_{}.pth'.format(args.version, epoch + 1, strftime("%a%d%b%Y-%H%M%S", gmtime()))
            torch.save(model_without_ddp.state_dict(), os.path.join(path_to_save, weight_name))
        
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    torch.save(model_without_ddp.state_dict(), os.path.join(path_to_save, weight_name)) 
                else:
                    print('eval ...')
                    model_eval = model_without_ddp

                    # set eval mode
                    model_eval.trainable = False
                    model_eval.eval()

                    # evaluate
                    eval_result_dict = evaluator.evaluate(model_eval)

                    if run is not None:
                        wandb.log({
                            "epoch": epoch+1,
                            "ap50_95": eval_result_dict['ap50_95'], 
                            "ap50": eval_result_dict['ap50'],
                            "ap75": eval_result_dict['ap75'],
                            "ap_normal": eval_result_dict['ap_normal'],
                            "ap_easy": eval_result_dict['ap_easy'],
                            "ap_medium": eval_result_dict['ap_medium'],
                            "ap_hard": eval_result_dict['ap_hard'],
                            "center_errors": eval_result_dict['center_errors'],
                            "center_errors_normal": eval_result_dict['center_errors_normal'],
                            "center_errors_easy": eval_result_dict['center_errors_easy'],
                            "center_errors_medium": eval_result_dict['center_errors_medium'],
                            "center_errors_hard": eval_result_dict['center_errors_hard'],
                        })

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, weight_name)) 

                    # set train mode.
                    model_eval.trainable = True
                    model_eval.train()
        
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

        # close mosaic augmentation
        if args.mosaic and max_epoch - epoch == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False
    
    wandb.finish()


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))

    if 'val_transform' in cfg.keys():
        val_transform = cfg['val_transform']
    else:
        val_transform = None

    train_transform = TrainTransforms(trans_config=trans_config,
                                    min_size=args.train_min_size,
                                    max_size=args.train_max_size,
                                    random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                    pixel_mean=cfg['pixel_mean'],
                                    pixel_std=cfg['pixel_std'],
                                    format=cfg['format'])
    val_transform = ValTransforms(min_size=args.val_min_size,
                                max_size=args.val_max_size,
                                pixel_mean=cfg['pixel_mean'],
                                pixel_std=cfg['pixel_std'],
                                trans_config=val_transform,
                                format=cfg['format'])
    color_augment = BaseTransforms(min_size=args.train_min_size,
                                max_size=args.train_max_size,
                                random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                pixel_mean=cfg['pixel_mean'],
                                pixel_std=cfg['pixel_std'],
                                format=cfg['format'])

    # dataset
    
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        # dataset
        dataset = VOCDetection(img_size=args.train_max_size,
                               data_dir=data_dir, 
                               transform=train_transform,
                               color_augment=color_augment,
                               mosaic=args.mosaic)
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'coco')
        num_classes = 80

        if args.train_ann_file is not None:
            dataset = COCODataset(img_size=args.train_max_size,
                                    data_dir=data_dir,
                                    image_set=args.train_img_folder,
                                    transform=train_transform,
                                    color_augment=color_augment,
                                    mosaic=args.mosaic,
                                    ann_file=args.train_ann_file)
            evaluator = COCOAPIEvaluator(data_dir=data_dir,EAD=cfg['EAD'],
                                        device=device,
                                        transform=val_transform,
                                        image_folder=args.val_img_folder,
                                        ann_file=args.val_ann_file)
        else:            

            # dataset
            dataset = COCODataset(img_size=args.train_max_size,
                                data_dir=data_dir,
                                image_set='train2017',
                                transform=train_transform,
                                color_augment=color_augment,
                                mosaic=args.mosaic)
            # evaluator
            evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                        device=device,
                                        transform=val_transform)
            
    elif args.dataset == 'coco_fb':
        data_dir = os.path.join(args.root, 'coco')
        num_classes = 2

        if args.train_ann_file is not None:
            dataset = COCODataset_FB(img_size=args.train_max_size,
                                    data_dir=data_dir,
                                    image_set=args.train_img_folder,
                                    image_folder=args.train_img_folder,
                                    transform=train_transform,
                                    color_augment=color_augment,
                                    mosaic=args.mosaic,
                                    ann_file=args.train_ann_file)
            evaluator = COCOAPIEvaluator(data_dir=data_dir,EAD=cfg['EAD'],
                                        device=device,
                                        transform=val_transform,
                                        image_folder=args.val_img_folder,
                                        ann_file=args.val_ann_file)
        else:            

            # dataset
            dataset = COCODataset(img_size=args.train_max_size,
                                data_dir=data_dir,
                                image_set='train2017',
                                transform=train_transform,
                                color_augment=color_augment,
                                mosaic=args.mosaic)
            # evaluator
            evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                        device=device,
                                        transform=val_transform)


    elif args.dataset == 'coco_fb_diff':
        data_dir = os.path.join(args.root, 'coco')
        num_classes = 2

        if args.train_ann_file is not None:
            
            if args.dense_guidance:
                
                source_trans_config = copy.deepcopy(trans_config)
                for t_id, trans in enumerate(source_trans_config):
                    if trans['name'] == 'EAD_Resize':
                        source_trans_config[t_id] = {'name': 'Resize'}
                
                print(f'SourceTransform: {source_trans_config}')
                
                source_transform = TrainTransforms(trans_config=source_trans_config,
                                    min_size=args.train_min_size,
                                    max_size=args.train_max_size,
                                    random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                    pixel_mean=cfg['pixel_mean'],
                                    pixel_std=cfg['pixel_std'],
                                    format=cfg['format'])
                
                assert args.source_img_folder is not None, 'source_img_folder is None'
                
                dataset = COCODataset_FB_Diff_withSource(img_size=args.train_max_size,
                                    data_dir=data_dir,
                                    image_set=args.train_img_folder,
                                    image_folder=args.train_img_folder,
                                    source_folder=args.source_img_folder,
                                    transform=train_transform,
                                    source_transform=source_transform,
                                    color_augment=color_augment,
                                    mosaic=args.mosaic,
                                    ann_file=args.train_ann_file)
            else:
            
                dataset = COCODataset_FB_Diff(img_size=args.train_max_size,
                                        data_dir=data_dir,
                                        image_set=args.train_img_folder,
                                        image_folder=args.train_img_folder,
                                        transform=train_transform,
                                        color_augment=color_augment,
                                        mosaic=args.mosaic,
                                        ann_file=args.train_ann_file)
            evaluator = COCOAPIEvaluator(data_dir=data_dir,EAD=cfg['EAD'],
                                        device=device,
                                        transform=val_transform,
                                        image_folder=args.val_img_folder,
                                        ann_file=args.val_ann_file)
        else:            

            # dataset
            dataset = COCODataset_FB_Diff(img_size=args.train_max_size,
                                data_dir=data_dir,
                                image_set='train2017',
                                transform=train_transform,
                                color_augment=color_augment,
                                mosaic=args.mosaic)
            # evaluator
            evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                        device=device,
                                        transform=val_transform)
            
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
                        )
    return dataloader

if __name__ == '__main__':
    train()