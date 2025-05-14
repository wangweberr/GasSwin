import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # pred 经过 sigmoid 激活
        pred = torch.sigmoid(pred)
        
        # 计算交集和并集
        intersection = (pred * target).sum(dim=(2, 3, 4))  # 时空维度(T,H,W)上求和
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
    
    #计算dice系数，用于评估性能
def calculate_dice(pred, target, smooth=1e-5):
    """计算 Dice 系数"""
    pred=pred.detach()#禁用梯度计算图，减少内存占用
    pred = torch.sigmoid(pred) > 0.5  # 转为二值
    pred = pred.float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

