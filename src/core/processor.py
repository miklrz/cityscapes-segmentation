import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torchvision.io import decode_image
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torchmetrics.segmentation import MeanIoU
from clearml import Task
from dataclasses import dataclass
from clearml import Task
import os


class Processor:
    def __init__(
        self,
        IMAGE_DATA_PATH,
        ANNOTATIONS_DATA_PATH,
        ANNOTATIONS_TYPES,
        ANNOTATIONS_PREFIX="_gtFine",
    ):
        self.IMAGE_DATA_PATH = IMAGE_DATA_PATH
        self.ANNOTATIONS_DATA_PATH = ANNOTATIONS_DATA_PATH
        self.ANNOTATIONS_PREFIX = ANNOTATIONS_PREFIX
        self.ANNOTATIONS_TYPES = ANNOTATIONS_TYPES

    def get_images(self):
        images = {}
        cities = os.listdir(self.IMAGE_DATA_PATH)
        for city in cities:
            city_image_path = os.path.join(self.IMAGE_DATA_PATH, city)
            files_image = os.listdir(city_image_path)
            for file in files_image:
                full_image_path = os.path.join(city_image_path, file)
                splitted = file.split("_")
                if len(splitted) != 4:
                    raise ValueError("Len of splitted != 4")
                image_type = "left"
                image_city = splitted[0]
                sequence_number = splitted[1]
                frame_number = splitted[2]
                image_name = f"{image_city}_{sequence_number}_{frame_number}"

                image_arr = images.get(image_name, {})
                image_arr.update({"left": full_image_path})
                for ANNOTATION_TYPE in self.ANNOTATIONS_TYPES:
                    annot_type = ANNOTATION_TYPE.split(".")[0]
                    image_arr.update(
                        {
                            annot_type: os.path.join(
                                self.ANNOTATIONS_DATA_PATH,
                                f"{image_city}/{image_name}{self.ANNOTATIONS_PREFIX}_{ANNOTATION_TYPE}",
                            )
                        }
                    )
                images.update({image_name: image_arr})

        for imgid in images.keys():
            if len(images[imgid]) != 5:
                raise ValueError("Len of arr %5 != 0")

        return images
