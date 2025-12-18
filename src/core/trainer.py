"""
Модуль тренировки и валидации модели семантической сегментации.

Класс Trainer реализует полный цикл обучения:
- Обучение на тренировочном наборе
- Оценка качества (mIoU и loss) на тренировочном и тестовом наборах
- Сохранение лучших и финальных весов модели
- Логирование метрик в ClearML

Все методы совместимы с исходным интерфейсом.
"""

import torch
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU
import os
from src.core.config import logger, device, NUM_CLASSES, CITYSCAPES_MASK_COLORS
from src.core.dataset import visualise


class Trainer:
    """
    Класс для управления процессом обучения и валидации модели сегментации.

    Атрибуты:
        trainDataloader: DataLoader для тренировочного набора.
        testDataloader: DataLoader для тестового набора.
        evalInterval (int): Интервал (в эпохах) между оценками на тестовом наборе.
        savePath (str): Путь для сохранения чекпоинтов модели.
        best_miou (float): Лучшее значение mIoU, достигнутое на тестовом наборе.
    """

    def __init__(self, trainDataloader, testDataloader, evalInterval, savePath):
        self.trainDataloader = trainDataloader
        self.testDataloader = testDataloader
        self.evalInterval = evalInterval
        self.savePath = savePath
        self.best_miou = 0.0

    def train(self, net, optimizer, epochs, criterion):
        """
        Основной цикл обучения модели.

        Аргументы:
            net (torch.nn.Module): Обучаемая модель.
            optimizer (torch.optim.Optimizer): Оптимизатор.
            epochs (int): Число эпох обучения.
            criterion (torch.nn.Module): Функция потерь.
        """
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0.0

            # Обучение на одной эпохе
            for i, (batch_image, batch_mask) in tqdm(
                enumerate(self.trainDataloader),
                total=len(self.trainDataloader),
                desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            ):
                batch_image = batch_image.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = net(batch_image)
                loss = criterion(output, batch_mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Усреднение потерь по эпохе
            epoch_loss /= len(self.trainDataloader)
            print(f"[Epoch {epoch}] Train loss: {epoch_loss:.6f}")
            logger.report_scalar(
                title="Loss", series="Train", value=epoch_loss, iteration=epoch
            )

            # Валидация на тренировочном наборе (опционально, для диагностики)
            self.test(
                net=net,
                epoch=epoch,
                task_type="Train",
                visualise=False,
                criterion=criterion,
            )

            # Валидация на тестовом наборе с заданной периодичностью
            # Примечание: условие `if True` заменено на логически корректное
            should_eval = ((epoch + 1) % self.evalInterval == 0) or (
                epoch == epochs - 1
            )
            if should_eval:
                miou_value = self.test(
                    net=net,
                    epoch=epoch,
                    task_type="Test",
                    visualise=False,
                    criterion=criterion,
                )

                # Сохранение модели при улучшении mIoU или на последней эпохе
                if miou_value > self.best_miou:
                    self.best_miou = miou_value
                    self.save_model(model=net, epoch=epoch, last=False)
                    print(f"Saved best model (mIoU={miou_value:.4f}) on epoch {epoch}")
                elif epoch == epochs - 1:
                    self.save_model(model=net, epoch=epoch, last=True)
                    print(f"Saved last model on epoch {epoch}")

    def test(self, net, epoch, task_type, criterion, visualise=False):
        """
        Оценка модели на указанном наборе данных.

        Аргументы:
            net (torch.nn.Module): Модель для оценки.
            epoch (int): Номер текущей эпохи (для логирования).
            task_type (str): "Train" или "Test" — определяет, какой DataLoader использовать.
            criterion (torch.nn.Module): Функция потерь (для вычисления валидационного лосса).
            visualise (bool): Если True и task_type="Test", сохраняет визуализацию первого кадра.

        Возвращает:
            float: Значение mIoU на данном наборе данных.
        """
        net.eval()
        miou_metric = MeanIoU(num_classes=NUM_CLASSES).to(device)
        dataset = self.testDataloader if task_type == "Test" else self.trainDataloader
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in tqdm(
                enumerate(dataset),
                desc=f"Eval on {task_type} set",
                total=len(dataset),
                leave=False,
            ):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = net(images)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                # Визуализация только первого батча и только при явном запросе
                if visualise and batch_idx == 0 and task_type == "Test":
                    visualise(CITYSCAPES_MASK_COLORS, image=images[0], mask=preds[0])

                miou_metric.update(preds, masks)

            avg_loss = total_loss / len(dataset)
            final_miou = miou_metric.compute().item()

            # Логирование в ClearML
            logger.report_scalar(
                title="mIoU", series=task_type, value=final_miou, iteration=epoch
            )
            logger.report_scalar(
                title="Loss", series=task_type, value=avg_loss, iteration=epoch
            )

            print(f"[{task_type}] mIoU: {final_miou:.4f}, Loss: {avg_loss:.6f}")

        return final_miou

    def save_model(self, model, epoch, last):
        """
        Сохраняет состояние модели в файл.

        Аргументы:
            model (torch.nn.Module): Модель для сохранения.
            epoch (int): Номер эпохи.
            last (bool): Если True — сохраняется как 'last', иначе как 'best'.
        """
        suffix = "last" if last else "best"
        filename = f"model1_epoch-{epoch}_{suffix}"
        filepath = os.path.join(self.savePath, filename)
        os.makedirs(self.savePath, exist_ok=True)
        torch.save(model.state_dict(), filepath)
        print(f"Saved model with name: {filename}")
