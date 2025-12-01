import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class CityscapesDataset(Dataset):
    def __init__(self, images: dict, keys=None, size=(256, 512)):
        self.images = images
        if not keys:
            self.keys = list(self.images.keys())
        else:
            self.keys = keys
        self.size = size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        image_index = self.keys[idx]
        image_dict = self.images[image_index]

        image_path = image_dict["left"]
        labelIds_path = image_dict["labelIds"]

        # mask_path = image_dict["color"]
        # instanceIds_path = image_dict["instanceIds"]
        # polygons_path = image_dict["polygons"]

        image = Image.open(image_path)
        mask = Image.open(labelIds_path)

        transform = transforms.Compose(
            [
                transforms.Resize(size=self.size, interpolation=Image.NEAREST),
            ]
        )
        image = transform(image)
        mask = transform(mask)

        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy(mask_array).long()
        image_tensor = TF.to_tensor(image)  # C,H,W

        return image_tensor, mask_tensor


def get_index_splits(images: dict):
    """
    Возвращает индексы train и test, которые передаются а датасет при создании
    """
    keys_list = list(images.keys())
    train_images, test_images = train_test_split(
        keys_list, test_size=0.2, random_state=42
    )
    return train_images, test_images


def visualise(CITYSCAPES_MASK_COLORS, image, mask):
    image, mask = image.cpu(), mask.cpu()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    def decode_segmap(mask, colormap=CITYSCAPES_MASK_COLORS):
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label in range(len(colormap)):
            num_true = (mask == label).sum().item()
            color_mask[mask == label] = colormap[label]
        return color_mask

    color_mask = decode_segmap(mask.numpy(), CITYSCAPES_MASK_COLORS)
    blended = (0.5 * image.permute(1, 2, 0).numpy() + 0.5 * (color_mask / 255.0)).clip(
        0, 1
    )

    axes[0].imshow(image.permute(1, 2, 0))
    axes[1].imshow(blended)
    plt.show()
