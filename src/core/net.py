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


"""
Реализация архитектуры U-Net для семантической сегментации.

U-Net — популярная encoder-decoder сеть с skip-connection'ами.
Преимущества перед SegNet:
- Лучшая передача пространственной информации через skip-connections
- Более стабильное обучение
- Часто даёт выше mIoU на тех же данных

Данная реализация:
- Сохраняет совместимость с интерфейсом SegNet
- Не использует Softmax на выходе (работает с логитами)
- Гарантирует совпадение размера выхода с входом при корректном размере входа (кратном 16)

Параметры аналогичны SegNet для лёгкой замены в main.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    """Два свёрточных слоя с BatchNorm и ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(x))
        return x


class UNet(nn.Module):
    """
    U-Net архитектура для семантической сегментации.

    Параметры:
        in_channels (int): Число входных каналов (обычно 3 для RGB).
        num_classes (int): Число классов сегментации.
        features (list[int]): Число каналов на каждом уровне. По умолчанию — оригинальная схема U-Net.

    Возвращает:
        torch.Tensor: Логиты формы [B, num_classes, H, W].
    """

    def __init__(self, in_channels=3, num_classes=32, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]  # как в оригинальной статье

        # Encoder (с downsampling'ом)
        self.enc1 = UNetConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = UNetConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetConvBlock(features[3], features[4])

        # Decoder (с upsampling'ом и skip-connections)
        self.upconv4 = nn.ConvTranspose2d(
            features[4], features[3], kernel_size=2, stride=2
        )
        self.dec4 = UNetConvBlock(
            features[4], features[3]
        )  # features[3] + features[3] от skip

        self.upconv3 = nn.ConvTranspose2d(
            features[3], features[2], kernel_size=2, stride=2
        )
        self.dec3 = UNetConvBlock(features[3], features[2])

        self.upconv2 = nn.ConvTranspose2d(
            features[2], features[1], kernel_size=2, stride=2
        )
        self.dec2 = UNetConvBlock(features[2], features[1])

        self.upconv1 = nn.ConvTranspose2d(
            features[1], features[0], kernel_size=2, stride=2
        )
        self.dec1 = UNetConvBlock(features[1], features[0])

        # Финальный классификатор
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов по методу Kaiming."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder с skip-connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Финальный выход
        out = self.final_conv(dec1)

        # Восстанавливаем исходный размер (на случай неточностей)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(
                out, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return out
