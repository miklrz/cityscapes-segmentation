import torch
from tqdm import tqdm
from torchmetrics.segmentation import MeanIoU
import os
from src.core.config import logger, device, NUM_CLASSES, CITYSCAPES_MASK_COLORS
from src.core.dataset import visualise


class Trainer:
    def __init__(self, trainDataloader, testDataloader, evalInterval, savePath):
        self.trainDataloader = trainDataloader
        self.testDataloader = testDataloader
        self.evalInterval = evalInterval
        self.savePath = savePath
        self.best_miou = 0.0

    def train(self, net, optimizer, epochs, criterion):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (batch_image, batch_mask) in tqdm(
                enumerate(self.trainDataloader), total=len(self.trainDataloader)
            ):
                batch_image, batch_mask = batch_image.to(device), batch_mask.to(device)
                optimizer.zero_grad()

                output = net(batch_image)

                loss = criterion(output, batch_mask)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            epoch_loss /= len(self.trainDataloader)
            print(f"train loss: {epoch_loss}")
            logger.report_scalar(
                title="Loss",
                series="Train",
                value=epoch_loss,
                iteration=epoch,
            )

            self.test(
                net=net,
                epoch=epoch,
                task_type="Train",
                visualise=False,
                criterion=criterion,
            )

            testCondition = (epoch + 1) % self.evalInterval == 0

            if True:
                miou = self.test(
                    net=net,
                    epoch=epoch,
                    task_type="Test",
                    visualise=False,
                    criterion=criterion,
                )
                if miou > self.best_miou:
                    self.save_model(model=net, epoch=epoch, last=False)
                    print(f"Saved best model on epoch {epoch}")
                elif epoch == epochs - 1:
                    self.save_model(model=net, epoch=epoch, last=True)
                    print(f"Saved last model")

            net.train()

    def test(self, net, epoch, task_type, criterion, visualise=False):
        net.eval()
        miou = MeanIoU(num_classes=NUM_CLASSES).to(device)
        dataset = self.testDataloader if task_type == "Test" else self.trainDataloader
        test_loss = 0.0

        with torch.no_grad():
            for _, (images, masks) in tqdm(
                enumerate(dataset),
                desc=f"Test on {task_type} dataset",
                total=len(dataset),
            ):
                images, masks = images.to(device), masks.to(device)
                outputs = net(images)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                if visualise and _ == 0:
                    visualise(CITYSCAPES_MASK_COLORS, image=images[0], mask=preds[0])
                miou.update(preds, masks)

            test_loss /= len(dataset)

            final_miou = miou.compute()
            logger.report_scalar(
                title="Miou",
                series=task_type,
                value=final_miou,
                iteration=epoch,
            )
            if task_type == "Test":
                logger.report_scalar(
                    title="Loss",
                    series=task_type,
                    value=final_miou,
                    iteration=epoch,
                )
            print(f"miou: {final_miou}")

        return miou

    def save_model(self, model, epoch, last):
        filename = f"model1_epoch-{epoch}_"
        if last:
            filename += "last"
        else:
            filename += "best"
        filepath = os.path.join(self.savePath, filename)
        torch.save(model.state_dict(), filepath)
        print(f"Saved model with name: {filename}")
