from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import io
import hashlib
from enum import Enum

app = FastAPI(
    title="Door & Window Detection API",
    description="Upload an image (.jpg or .png) to detect doors and windows. Select response type: JSON or image.",
    version="1.0.0"
)

model = None
executor = None

class ResponseType(str, Enum):
    empty = ""
    json = "json"
    image = "image"

def load_model():
    global model
    model = YOLO("yolov8m_custom.pt")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_image, verbose=False, conf=0.5, iou=0.45)

@app.on_event("startup")
async def startup_event():
    global executor
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_model)

@app.on_event("shutdown")
async def shutdown_event():
    global executor
    if executor:
        executor.shutdown(wait=True)

def generate_color_for_label(label: str) -> tuple:
    hash_value = int(hashlib.md5(label.encode()).hexdigest(), 16)
    np.random.seed(hash_value % (2**32))
    color = tuple(int(c) for c in np.random.randint(0, 256, 3))
    return color

def process_image_sync(image_bytes: bytes) -> dict:
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    max_size = 1280

    if max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    results = model.predict(
        image, 
        verbose=False,
        conf=0.5,
        iou=0.45,
        max_det=100,
        half=torch.cuda.is_available()
    )

    detections = []
    result = results[0]
    if result.boxes:
        for i in range(len(result.boxes)):
            label = model.names[int(result.boxes.cls[i])]
            confidence = float(result.boxes.conf[i])
            if confidence >= 0.5:
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
                color = generate_color_for_label(label)

                cv2.rectangle(image, pt1, pt2, color, 2)
                label_text = f"{label}: {confidence:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image, (int(x1), int(y1) - text_size[1] - 4),
                              (int(x1) + text_size[0], int(y1)), color, -1)
                cv2.putText(image, label_text, (int(x1), int(y1) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": [round(coord, 2) for coord in bbox]
                })

    pil_result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    pil_result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return {"detections": detections, "image": img_byte_arr.getvalue()}

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Upload a JPG or PNG image"),
    response_type: ResponseType = Query(..., description="Select response type: 'json' or 'image'")
):
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_image_sync, content)

    if response_type == ResponseType.image:
        return StreamingResponse(
            io.BytesIO(result["image"]),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=detected_objects.png",
                "X-Detection-Count": str(len(result["detections"]))
            }
        )
    else:
        return JSONResponse(
            content={"detections": result["detections"]},
            headers={"X-Detection-Count": str(len(result["detections"]))}
        )
