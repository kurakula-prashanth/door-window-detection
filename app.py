from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(
    title="Door & Window Detection API",
    description="Upload an image (.jpg or .png) to detect doors and windows.",
    version="1.0.0"
)

# Global variables for model and thread pool
model = None
executor = None

def load_model():
    """Load and optimize the YOLO model"""
    global model
    model = YOLO("yolov8m_custom.pt")
    
    # Optimize model for inference
    if torch.cuda.is_available():
        model.to('cuda')
        print("Model loaded on GPU")
    else:
        model.to('cpu')
        print("Model loaded on CPU")
    
    # Warm up the model with a dummy prediction
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_image, verbose=False, conf=0.5, iou=0.45)
    print("Model warmed up successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize model and thread pool on startup"""
    global executor
    # Create thread pool for CPU-bound tasks
    executor = ThreadPoolExecutor(max_workers=2)
    
    # Load model in a separate thread to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_model)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global executor
    if executor:
        executor.shutdown(wait=True)

def process_image_sync(image_bytes: bytes) -> dict:
    """Synchronous image processing function"""
    try:
        # Use PIL for faster image loading
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize image if too large (maintain aspect ratio)
        height, width = image.shape[:2]
        max_size = 1280  # Reduce from default to speed up inference
        
        if max(height, width) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Run inference with optimized parameters
        results = model.predict(
            image, 
            verbose=False,
            conf=0.5,  # Confidence threshold
            iou=0.45,  # IoU threshold for NMS
            max_det=100,  # Maximum detections
            half=True if torch.cuda.is_available() else False  # Use FP16 if on GPU
        )
        
        # Process results efficiently
        detections = []
        if results and len(results) > 0:
            result = results[0]  # Get first result
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    label = model.names[int(boxes.cls[i])]
                    confidence = float(boxes.conf[i])
                    
                    # Only include high-confidence detections
                    if confidence >= 0.5:
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                        
                        detections.append({
                            "label": label,
                            "confidence": round(confidence, 2),
                            "bbox": [round(coord, 2) for coord in bbox]
                        })
        
        return {"detections": detections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def redirect_to_docs():
    """Redirect to Swagger UI for easy testing"""
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Optimized prediction endpoint"""
    # Validate file extension
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, or PNG images are allowed"
        )
    
    # Check file size (limit to 10MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum 10MB allowed."
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )
    
    # Process image in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, process_image_sync, content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))