# src/api/app.py
import os
import tempfile
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
from src.core.config import device, NUM_CLASSES, config
from src.core.net import SegNet
from src.core.inference import process_frame
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Semantic Segmentation API")

model = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
model_path = Path(config.SAVE_PATH) / "best_model.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Train it first!")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
logger.info("Model loaded successfully.")


@app.post("/segment_video")
async def segment_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".mp4", "_segmented.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open input video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            segmented_frame = process_frame(model, frame, (height, width))
            out.write(segmented_frame)
    finally:
        cap.release()
        out.release()
        os.unlink(input_path)

    def iterfile():
        with open(output_path, mode="rb") as f:
            yield from f
        os.remove(output_path)

    return StreamingResponse(iterfile(), media_type="video/mp4")
