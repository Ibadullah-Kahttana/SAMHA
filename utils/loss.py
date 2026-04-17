import torch.nn as nn
import torch.nn.functional as F
import torch

def cross_entropy(input, target, class_weights=None, reduction="mean", ignore_index=-100):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=input.dtype, device=input.device)

    loss = F.cross_entropy(
        input,
        target,
        weight=weight_tensor,
        reduction=reduction,
        ignore_index=ignore_index,
    )

    return loss
    
def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view) 
    ones = 1.
    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=None, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore
        self.class_weights = class_weights

    def forward(self, input, target):
        '''
        Focal Loss implementation
        Supports ignore_index and optional class weights
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, input.size(1))
        
        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.class_weights is not None:
            class_idx = target.argmax(dim=1) if len(target.shape) > 1 else target
            weights = torch.FloatTensor(self.class_weights).cuda()[class_idx]
            batch_loss = batch_loss * weights

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class SegmentationCELoss(torch.nn.Module):
    def __init__(self, class_weights=None, ignore_index=0, reduction="mean"):
        super().__init__()
        self.register_buffer("weight",
            None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32))
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        if self.weight is not None:
            w = self.weight.to(device=logits.device, dtype=logits.dtype)
        else:
            w = None
        return F.cross_entropy(logits, target, weight=w, ignore_index=self.ignore_index, reduction=self.reduction)


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] model output (raw logits)
            targets: [B, H, W] ground truth labels (class indices)
        
        Returns:
            dice_loss: scalar tensor
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        targets_one_hot = F.one_hot(targets, self.num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        probs_flat = probs.reshape(-1)  # [B*C*H*W]
        targets_flat = targets_one_hot.reshape(-1)  # [B*C*H*W]
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class FocalDiceComboLoss(nn.Module):
    def __init__(self, num_classes=2, gamma=2, lambda_dice=0.4, class_weights=None, smooth=1e-7):
        super(FocalDiceComboLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_dice = lambda_dice
        self.smooth = smooth
        self.dice_loss = DiceLoss(num_classes=num_classes, smooth=smooth)
        self.focal_loss = FocalLoss(gamma=gamma, class_weights=class_weights)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] model output (raw logits)
            targets: [B, H, W] ground truth labels (class indices)
        
        Returns:
            total_loss: scalar tensor
        """
        dice_loss_val = self.dice_loss(logits, targets)
        focal_loss_val = self.focal_loss(logits, targets)
        total_loss = self.lambda_dice * dice_loss_val + (1 - self.lambda_dice) * focal_loss_val
        return total_loss

