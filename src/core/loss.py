import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        smooth = 1

        target_one_hot = nn.functional.one_hot(
            target.long(), num_classes=self.num_classes
        )
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3))
        pred_sum = torch.sum(pred, dim=(0, 2, 3))
        target_sum = torch.sum(target_one_hot, dim=(0, 2, 3))

        dice = 1 - ((2.0 * intersection + smooth) / (pred_sum + target_sum + smooth))
        return dice.mean()
