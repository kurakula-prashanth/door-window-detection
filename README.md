# Door & Window Detection using YOLOv8

A custom-trained YOLOv8 model for detecting doors and windows in construction blueprint-style images, deployed as a FastAPI service.

## 🚀 Demo

**Live API**: [https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection](https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection)

**GitHub Repository**: [https://github.com/kurakula-prashanth/door-window-detection](https://github.com/kurakula-prashanth/door-window-detection)

## 📋 Project Overview

This project implements a complete machine learning pipeline for detecting doors and windows in architectural blueprints:

1. **Manual Data Labeling** - Created custom dataset with bounding box annotations
2. **Model Training** - Trained YOLOv8 model from scratch using only custom-labeled data
3. **API Development** - Built FastAPI service for real-time inference
4. **Deployment** - Deployed to Hugging Face Spaces with Docker

## 🎯 Classes Detected

- `door` - Door symbols in blueprints
- `window` - Window symbols in blueprints

## 🛠️ Setup & Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/kurakula-prashanth/door-window-detection.git
cd door-window-detection
```

2. **Create virtual environment**
```bash
python3.12 -m venv yolo8_custom
source yolo8_custom/bin/activate  # On Windows: yolo8_custom\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API locally**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Training Process

### Step 1: Data Labeling
- Used **LabelImg** for manual annotation
- Labeled 15-20 construction blueprint images
- Created bounding boxes for doors and windows only
- Generated YOLO format labels (.txt files)

### Step 2: Model Training
```bash
yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8m.pt imgsz=640
```

**Training Configuration:**
- Base Model: YOLOv8 Medium (yolov8m.pt)
- Epochs: 100
- Image Size: 640x640
- Classes: 2 (door, window)

### Step 3: Model Testing
```bash
yolo task=detect mode=predict model=best.pt show=true conf=0.5 source=12.png line_thickness=1
```

## 🔌 API Usage

### Endpoint
```
POST /predict
```

### Request
- **Content-Type**: `multipart/form-data`
- **File**: Upload PNG or JPG image (max 10MB)

### Response Format
```json
{
  "detections": [
    {
      "label": "door",
      "confidence": 0.91,
      "bbox": [x, y, width, height]
    },
    {
      "label": "window", 
      "confidence": 0.84,
      "bbox": [x, y, width, height]
    }
  ]
}
```

### cURL Example
```bash
curl -X POST "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict" \
     -F "file=@your_blueprint.png"
```

### Python Example
```python
import requests

url = "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict"
files = {"file": open("blueprint.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🐳 Docker Deployment

The application is containerized using Docker:

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## 📦 Dependencies

```txt
fastapi
uvicorn
ultralytics
opencv-python-headless
pillow
torch
numpy
python-multipart
```

## ⚡ Performance Features

- **GPU Acceleration**: Automatically uses CUDA if available
- **Model Optimization**: FP16 precision on GPU for faster inference
- **Async Processing**: Non-blocking image processing with ThreadPoolExecutor
- **Image Preprocessing**: Automatic resizing and format conversion
- **Memory Efficient**: Optimized for production deployment

## 📁 Project Structure

```
door-window-detection/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── yolov8m_custom.pt     # Trained model weights
├── data_custom.yaml      # Training configuration
├── classes.txt           # Class names
├── datasets/             # Training data
│   ├── images/
│   └── labels/
└── README.md            # This file
```

## 🔍 Model Details

- **Architecture**: YOLOv8 Medium
- **Input Size**: 640x640 pixels
- **Classes**: 2 (door, window)
- **Confidence Threshold**: 0.5
- **IoU Threshold**: 0.45
- **Training Data**: Custom-labeled blueprint images

## 📈 Results & Screenshots

### Training Progress
- Loss curves and training metrics
- Model performance on validation set
- Convergence after 100 epochs

### Labeling Process
- LabelImg interface screenshots
- Sample .txt label files
- Annotation examples

### API Testing
- Live API responses
- Detection visualizations
- Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI framework
- Hugging Face Spaces for deployment
- LabelImg for annotation tool

## 📞 Contact

**Developer**: Kurakula Prashanth  
**GitHub**: [@kurakula-prashanth](https://github.com/kurakula-prashanth)  
**Project Link**: [https://github.com/kurakula-prashanth/door-window-detection](https://github.com/kurakula-prashanth/door-window-detection)
