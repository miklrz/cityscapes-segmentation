import torch
from clearml import Task
from dataclasses import dataclass
from clearml import Task
import os


@dataclass
class Config:
    DATA_PATH: str = "/home/hxastur/vscode_projects/cityscapes-segmentation/dataset"
    ANNOTATIONS_DATA_PATH: str = os.path.join(DATA_PATH, "gtFine/gtFine")
    ANNOTATIONS_TRAIN_PATH: str = os.path.join(ANNOTATIONS_DATA_PATH, "train")
    ANNOTATIONS_TEST_PATH: str = os.path.join(ANNOTATIONS_DATA_PATH, "test")
    ANNOTATIONS_VAL_PATH: str = os.path.join(ANNOTATIONS_DATA_PATH, "val")
    ANNOTATION_TYPES = ["color.png", "instanceIds.png", "labelIds.png", "polygons.json"]
    IMAGE_DATA_PATH: str = os.path.join(DATA_PATH, "left/leftImg8bit")
    IMAGE_TRAIN_PATH: str = os.path.join(IMAGE_DATA_PATH, "train")
    IMAGE_TEST_PATH: str = os.path.join(IMAGE_DATA_PATH, "test")
    IMAGE_VAL_PATH: str = os.path.join(IMAGE_DATA_PATH, "val")
    IMAGE_TYPE: str = "leftImg8bit"
    ANNOTATIONS_PREFIX: str = "gtFine"
    SAVE_PATH: str = (
        "/home/hxastur/vscode_projects/cityscapes-segmentation/saved_models"
    )
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 10
    IMAGE_SIZE = (128, 256)
    evalInterval = 1
    clearml_project_name = "Cityscapes_segmentation"
    clearml_task_name = "segnet_baseline_1"


config = Config()
task = Task.init(project_name="Cityscapes_segmentation", task_name="segnet_baseline_1")
task.connect(config)
logger = task.get_logger()
device = "cuda" if torch.cuda.is_available else "spu"
NUM_CLASSES = 34

CITYSCAPES_MASK_CLASSES = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}

CITYSCAPES_MASK_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
}
