"""
Реализация архитектуры SegNet для семантической сегментации изображений.

SegNet — это encoder-decoder сеть, предложенная в 2015 году.
Особенность: при пулинге сохраняются индексы максимальных значений,
а при апсемплинге они используются для точного восстановления пространственной структуры.

Данная реализация:
- Не включает Softmax на выходе (ожидается использование с лоссами, работающими с логитами).
- Гарантирует совпадение размера выхода с размером входа через билинейную интерполяцию.
- Поддерживает произвольное число классов и гибкую настройку каналов.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """
    Базовый блок свёртки: Conv2d → BatchNorm2d → ReLU.

    Параметры:
        in_channels (int): Число входных каналов.
        out_channels (int): Число выходных каналов.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # Сохраняем размер изображения
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True экономит память

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через блок.

        Аргументы:
            x (torch.Tensor): Входной тензор формы [B, C_in, H, W].

        Возвращает:
            torch.Tensor: Выходной тензор формы [B, C_out, H, W].
        """
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """
    Блок энкодера SegNet: последовательность CNNBlock + MaxPool2d с сохранением индексов.

    Параметры:
        in_channels (int): Число входных каналов.
        out_channels (int): Число выходных каналов.
        num_convs (int): Количество свёрточных блоков (2 или 3).
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super(EncoderBlock, self).__init__()
        assert num_convs in (2, 3), "Число свёрток должно быть 2 или 3."

        layers = [CNNBlock(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            layers.append(CNNBlock(out_channels, out_channels))
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход энкодера.

        Аргументы:
            x (torch.Tensor): Входной тензор [B, C_in, H, W].

        Возвращает:
            tuple: (выход после пула [B, C_out, H/2, W/2], индексы пула).
        """
        x = self.convs(x)
        pooled, indices = self.pool(x)
        return pooled, indices


class DecoderBlock(nn.Module):
    """
    Блок декодера SegNet: MaxUnpool2d + последовательность CNNBlock.

    Параметры:
        in_channels (int): Число входных каналов.
        out_channels (int): Число выходных каналов.
        num_convs (int): Количество свёрточных блоков (2 или 3).
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super(DecoderBlock, self).__init__()
        assert num_convs in (2, 3), "Число свёрток должно быть 2 или 3."

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        layers = [CNNBlock(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            layers.append(CNNBlock(out_channels, out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход декодера.

        Аргументы:
            x (torch.Tensor): Входной тензор [B, C_in, H, W].
            indices (torch.Tensor): Индексы от соответствующего пула энкодера.

        Возвращает:
            torch.Tensor: Выходной тензор [B, C_out, H*2, W*2].
        """
        x = self.unpool(x, indices)
        x = self.convs(x)
        return x


class SegNet(nn.Module):
    """
    Полная архитектура SegNet для семантической сегментации.

    Структура по умолчанию соответствует оригинальной статье:
    - Энкодер: 5 блоков с количеством каналов [64, 128, 256, 512, 512]
    - Первые два блока: по 2 свёртки, остальные — по 3

    Параметры:
        in_channels (int): Число каналов входного изображения (обычно 3 для RGB).
        num_classes (int): Число классов для сегментации (включая фон).
        channels (list[int]): Список числа каналов на каждом уровне энкодера.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 32,
        channels: list[int] = None,
    ):
        super(SegNet, self).__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 512]  # Оригинальная конфигурация

        assert len(channels) == 5, "SegNet требует ровно 5 уровней энкодера/декодера."

        # === Энкодер ===
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for i, ch in enumerate(channels):
            num_convs = 2 if i < 2 else 3
            self.encoders.append(EncoderBlock(prev_ch, ch, num_convs))
            prev_ch = ch

        # === Декодер ===
        self.decoders = nn.ModuleList()
        # Декодер использует обратную последовательность каналов
        decoder_channels = list(reversed(channels))
        for i, ch in enumerate(decoder_channels):
            if i == len(decoder_channels) - 1:
                out_ch = num_classes  # Последний слой — число классов
            else:
                out_ch = decoder_channels[i + 1]
            num_convs = 2 if i < 2 else 3
            self.decoders.append(DecoderBlock(ch, out_ch, num_convs))

        # Инициализация весов (опционально, но рекомендуется)
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов по методу Kaiming (He) для ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход всей сети.

        Аргументы:
            x (torch.Tensor): Входное изображение [B, in_channels, H, W].
                Рекомендуется, чтобы H и W делились на 32.

        Возвращает:
            torch.Tensor: Логиты сегментации [B, num_classes, H, W].
                Размер совпадает с входом благодаря финальной интерполяции.
        """
        original_size = x.shape[2:]  # (H, W)
        pool_indices = []

        # --- Энкодер ---
        for encoder in self.encoders:
            x, indices = encoder(x)
            pool_indices.append(indices)

        # --- Декодер ---
        for i, decoder in enumerate(self.decoders):
            # Используем индексы в обратном порядке
            indices = pool_indices[-(i + 1)]
            x = decoder(x, indices)

        # --- Восстановление исходного размера ---
        # Защищает от несоответствия размеров из-за особенностей MaxUnpool
        if x.shape[2:] != original_size:
            x = F.interpolate(
                x, size=original_size, mode="bilinear", align_corners=False
            )

        # ВАЖНО: Softmax НЕ применяется! Лосс работает с логитами.
        return x
