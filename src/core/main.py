from torch.utils.data import DataLoader
from torch.optim import Adam
from src.core.config import config, device, NUM_CLASSES
from src.core.processor import Processor
from src.core.dataset import get_index_splits, CityscapesDataset
from src.core.trainer import Trainer
from src.core.net import SegNet
from src.core.loss import DiceLoss


def execute_training():

    processor = Processor(
        IMAGE_DATA_PATH=config.IMAGE_TRAIN_PATH,
        ANNOTATIONS_DATA_PATH=config.ANNOTATIONS_TRAIN_PATH,
        ANNOTATIONS_TYPES=config.ANNOTATION_TYPES,
    )
    images = processor.get_images()
    train_images_idx, test_images_idx = get_index_splits(images)

    trainDataset = CityscapesDataset(images, train_images_idx)
    testDataset = CityscapesDataset(images, test_images_idx)

    trainDataloader = DataLoader(dataset=trainDataset, batch_size=config.batch_size)
    testDataloader = DataLoader(dataset=testDataset, batch_size=1)

    trainer = Trainer(
        trainDataloader=trainDataloader,
        testDataloader=testDataloader,
        evalInterval=config.evalInterval,
        savePath=config.SAVE_PATH,
    )

    net = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    optimizer = Adam(net.parameters(), lr=config.learning_rate)
    criterion = DiceLoss(NUM_CLASSES)

    trainer.train(
        net=net, optimizer=optimizer, epochs=config.epochs, criterion=criterion
    )


if __name__ == "__main__":
    execute_training()
