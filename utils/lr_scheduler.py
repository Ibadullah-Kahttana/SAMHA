import math
import torch
import torch.optim as optim

class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
        self.mode = mode
        self.lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        print(f'Using {self.mode} LR Scheduler!')
        if mode == 'step':
            assert lr_step

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError(f"LR Scheduler mode '{self.mode}' not implemented")
        
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        
        if epoch > self.epoch:
            print(f'\n=>Epoch {epoch}, LR = {lr:.7f}, Best = {best_pred:.4f}')
            self.epoch = epoch
        
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0:
                    optimizer.param_groups[i]['lr'] = lr

def get_optimizer_and_scheduler(model, base_learning_rate, num_epochs, iters_per_epoch):
    print(f"\n --- Configuring Optimizer & Scheduler \n")    
    learning_rate = base_learning_rate if base_learning_rate > 0 else 2e-4
    print(f"Learning Rate: {learning_rate}")
    
    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    optimizer = torch.optim.AdamW(
        _unwrap(model).parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    print(f"✓ Optimizer: AdamW (weight_decay=0.01)")

    # cos, poly, step
    scheduler = LR_Scheduler(
        'poly', 
        learning_rate, 
        num_epochs, 
        iters_per_epoch, 
        warmup_epochs=10
    )
    print(f"✓ Scheduler: Poly (warmup=10 epochs)")

    return optimizer, scheduler