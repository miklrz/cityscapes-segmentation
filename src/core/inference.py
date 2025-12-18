# src/core/inference.py
import torch
import cv2
import numpy as np
from torchvision import transforms
from src.core.config import device, CITYSCAPES_MASK_COLORS


def preprocess_frame(frame):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),  # или как у тебя в датасете
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(frame).unsqueeze(0)  # (1, C, H, W)


def decode_segmap(label_mask):
    """Convert label mask to color image using CITYSCAPES_MASK_COLORS"""
    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    for ll in range(len(CITYSCAPES_MASK_COLORS)):
        r[label_mask == ll] = CITYSCAPES_MASK_COLORS[ll][0]
        g[label_mask == ll] = CITYSCAPES_MASK_COLORS[ll][1]
        b[label_mask == ll] = CITYSCAPES_MASK_COLORS[ll][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def process_frame(model, frame, original_size):
    """Run inference on a single frame and return overlaid color segmentation"""
    with torch.no_grad():
        input_tensor = preprocess_frame(frame).to(device)
        output = model(input_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # (H, W)

    # Resize prediction to original frame size
    pred_resized = cv2.resize(
        pred.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    color_mask = decode_segmap(pred_resized)
    overlay = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    # Либо просто верни color_mask, если не нужен оверлей
    return color_mask
