"""
Реализация многоклассового Dice Loss для семантической сегментации.

Dice Loss особенно эффективен при работе с несбалансированными классами.
В отличие от CrossEntropy, он напрямую оптимизирует метрику сегментации (Dice/F1).

Важно:
- Модель должна возвращать **сырые логиты** (без Softmax на выходе).
- Эта реализация применяет Softmax внутри, чтобы получить вероятности.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Многоклассовый Dice Loss.

    Параметры:
        num_classes (int): Число классов сегментации.
        ignore_index (int, optional): Индекс класса, который игнорируется при подсчёте.
            По умолчанию None (все классы учитываются).
        smooth (float): Числитель сглаживания для предотвращения деления на ноль.
        eps (float): Малое значение для численной стабильности в знаменателе.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = None,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет Dice Loss.

        Аргументы:
            pred (torch.Tensor): Сырые логиты от модели, форма [B, C, H, W].
            target (torch.Tensor): Целевые метки, форма [B, H, W] (тип: long/int64).

        Возвращает:
            torch.Tensor: Скаляр — усреднённый по классам Dice Loss.
        """
        # Применяем softmax, чтобы получить вероятности
        pred_probs = F.softmax(pred, dim=1)  # [B, C, H, W]

        # Преобразуем целевые метки в one-hot
        target_one_hot = F.one_hot(
            target.long(), num_classes=self.num_classes
        )  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Если задан ignore_index, маскируем соответствующие пиксели
        if self.ignore_index is not None:
            # Создаём маску игнорируемых пикселей
            valid_mask = (target != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
            pred_probs = pred_probs * valid_mask
            target_one_hot = target_one_hot * valid_mask

        # Вычисляем числитель и знаменатель Dice по каждому классу
        # Суммируем по пространственным измерениям (H, W) и батчу (B)
        intersection = torch.sum(pred_probs * target_one_hot, dim=(0, 2, 3))  # [C]
        pred_sum = torch.sum(pred_probs, dim=(0, 2, 3))  # [C]
        target_sum = torch.sum(target_one_hot, dim=(0, 2, 3))  # [C]

        # Dice-коэффициент для каждого класса: (2*|X∩Y| + smooth) / (|X| + |Y| + smooth)
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.eps
        )

        # Усредняем по всем классам и преобразуем в loss: 1 - dice
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss
