
# Build warmup scheduler

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# https://kikaben.com/transformers-training-details/
class Transformer_WarmUpScheduler(_LRScheduler):
    def __init__(self, 
                 args,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        
        if hasattr(args, 'reduce_steps'):
            self.reduce_steps = args.reduce_steps
        
        self.num_param_groups = len(optimizer.param_groups)
        self.args = args

        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        
        factor = 1.0
        
        if hasattr(self, 'reduce_steps') and self._step_count > self.reduce_steps:
            factor = 0.1
        
        if self.num_param_groups == 3:
            return [lr * factor * self.args.model_lr, lr * factor * self.args.yoloh_lr, lr * factor * self.args.bk_lr]
        elif self.num_param_groups == 2:
            return [lr * factor  * self.args.model_lr, lr * factor * self.args.yoloh_lr]
        else:
            return [lr * factor ] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

def build_scheduler(args, optimizer, cfg, **kwargs):
    
    if 'scheduler_cfg' not in cfg.keys():
        warmup_scheduler = build_transformer_warmup(args, optimizer, **kwargs)
    
    else:
        if cfg['scheduler_cfg']['name'] == 'transformer_warmup':
            warmup_scheduler = build_transformer_warmup(args, optimizer, **kwargs)
        elif cfg['scheduler_cfg']['name'] == 'linear_yoloh':
            
            total_steps = kwargs['total_steps']
            max_epoch = cfg['epoch'][args.schedule]['max_epoch']
            steps_per_epoch = total_steps // max_epoch
            milestones = [int(steps_per_epoch * x) for x in cfg['epoch'][args.schedule]['lr_epoch']]
            
            warmup_scheduler = Linear_MultiStep_WarmUpScheduler(optimizer=optimizer,
                                            base_lr=args.base_lr,
                                            wp_iter=cfg['wp_iter'],
                                            milestones=milestones,
                                            warmup_factor=cfg['warmup_factor'])
        else:
            raise ValueError('Unknown scheduler: {}'.format(cfg['scheduler_cfg']['name']))
        
    return warmup_scheduler
    


def build_transformer_warmup(args, optimizer, total_steps, dim_embed, warmup_steps=None, **kwargs):
    if warmup_steps is not None:
        warmup_scheduler = Transformer_WarmUpScheduler(args, optimizer, dim_embed, warmup_steps)
    else:
        warmup_scheduler = Transformer_WarmUpScheduler(args, optimizer, dim_embed, int(total_steps*0.2))
    return warmup_scheduler


def build_warmup(name='linear', 
                 base_lr=0.01, 
                 wp_iter=500, 
                 warmup_factor=0.00066667):
    print('==============================')
    print('WarmUpScheduler: {}'.format(name))
    print('--base_lr: {}'.format(base_lr))
    print('--warmup_factor: {}'.format(warmup_factor))
    print('--wp_iter: {}'.format(wp_iter))

    warmup_scheduler = WarmUpScheduler(name=name, 
                                       base_lr=base_lr, 
                                       wp_iter=wp_iter, 
                                       warmup_factor=warmup_factor)
    
    return warmup_scheduler


class Linear_MultiStep_WarmUpScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 base_lr: float,
                 wp_iter: int,
                 warmup_factor: float,
                 milestones: list,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor
        self.milestones = milestones

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        
        if self._step_count < self.wp_iter:
            # warmup
            alpha = self._step_count / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            tmp_lr = self.base_lr * warmup_factor
            
            lr = []
            
            for param_group in self.optimizer.param_groups:
                
                init_lr = param_group['initial_lr']
                ratio = init_lr / self.base_lr

                lr.append(tmp_lr * ratio)
                
        elif self._step_count in self.milestones:
            # print(self.optimizer.param_groups[0]['lr'])
            lr = [param['lr'] * 0.1 for param in self.optimizer.param_groups]
            
        else:
            # print(self.optimizer.param_groups[0]['lr'])
            lr = [param['lr'] for param in self.optimizer.param_groups]
        
        return lr

                           
# Basic Warmup Scheduler
class WarmUpScheduler(object):
    def __init__(self, 
                 name='linear', 
                 base_lr=0.01, 
                 wp_iter=500, 
                 warmup_factor=0.00066667):
        self.name = name
        self.base_lr = base_lr
        self.wp_iter = wp_iter
        self.warmup_factor = warmup_factor


    def set_lr(self, optimizer, lr, base_lr):
        for param_group in optimizer.param_groups:
            init_lr = param_group['initial_lr']
            ratio = init_lr / base_lr
            param_group['lr'] = lr * ratio


    def warmup(self, iter, optimizer):
        # warmup
        assert iter < self.wp_iter
        if self.name == 'exp':
            tmp_lr = self.base_lr * pow(iter / self.wp_iter, 4)
            self.set_lr(optimizer, tmp_lr, self.base_lr)

        elif self.name == 'linear':
            alpha = iter / self.wp_iter
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            tmp_lr = self.base_lr * warmup_factor
            self.set_lr(optimizer, tmp_lr, self.base_lr)


    def __call__(self, iter, optimizer):
        self.warmup(iter, optimizer)
        