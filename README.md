# Door & Window Detection using YOLOv8

A custom-trained YOLOv8 model for detecting doors and windows in construction blueprint-style images, deployed as a FastAPI service with dual response modes.

## 🚀 Demo

**Live API**: [https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection](https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection)

**GitHub Repository**: [https://github.com/kurakula-prashanth/door-window-detection](https://github.com/kurakula-prashanth/door-window-detection)

## 📋 Project Overview

This project implements a complete machine learning pipeline for detecting doors and windows in architectural blueprints:

1. **Manual Data Labeling** - Created custom dataset with bounding box annotations
2. **Model Training** - Trained YOLOv8 model from scratch using only custom-labeled data
3. **API Development** - Built FastAPI service with dual response modes (JSON + annotated images)
4. **Deployment** - Deployed to Hugging Face Spaces with Docker

## 🎯 Classes Detected

- `door` - Door symbols in blueprints
- `window` - Window symbols in blueprints

## ✨ Key Features

- **Dual Response Modes**: Get JSON data or annotated images
- **Interactive Swagger UI**: Built-in API documentation at `/docs`
- **Smart Image Processing**: Automatic resizing for large images (max 1280px)
- **GPU Acceleration**: CUDA support with FP16 precision
- **Async Processing**: Non-blocking inference with ThreadPoolExecutor
- **Dynamic Color Coding**: Consistent colors for each detection class
- **Confidence Filtering**: Configurable confidence thresholds (default: 0.5)

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

5. **Access the API**
- **Interactive Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000/predict

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
![Labeling image using labelImg - 1](https://github.com/user-attachments/assets/609fd6ee-fcc7-4c6a-973b-6c539e8515c5)
![Labeling image using labelImg - 2](https://github.com/user-attachments/assets/3666c451-8bc4-4d57-9ffa-48611deca6d3)
![Labeling image using labelImg - 3](https://github.com/user-attachments/assets/2f5f23fb-1086-412f-82a1-f1ff5e24dd75)
![Labeling image using labelImg - 4](https://github.com/user-attachments/assets/8bccf20e-d5dc-4d1b-923b-7f603d64f5d2)

**Training Configuration:**
- Base Model: YOLOv8 Medium (yolov8m.pt)
- Epochs: 100
- Image Size: 640x640
- Classes: 2 (door, window)

![Training_img 1](https://github.com/user-attachments/assets/91d56bd7-ad51-412a-ac1d-f6519f4fb192)
![Training_img 2](https://github.com/user-attachments/assets/2c7e39c2-62ff-42ed-8f36-8246a1ef6754)
![Training_img 3](https://github.com/user-attachments/assets/334426cf-1189-45cc-a8a0-1fa5e17b7054)
![Training_img 4](https://github.com/user-attachments/assets/2a6b04e2-e7c3-476f-9490-f60725312eb4)

### Step 3: Model Testing
```bash
yolo task=detect mode=predict model=best.pt show=true conf=0.5 source=12.png line_thickness=1
```
![Testing_img 1](https://github.com/user-attachments/assets/3be7fed0-f8a0-4844-b203-d649fe93144a)
![Testing_img 2](https://github.com/user-attachments/assets/d1069eac-8e16-47c4-88a1-0d9707b81b75)

## 🔌 API Usage

### Main Endpoint
```
POST /predict
```

### Parameters
- **file** (required): Upload PNG or JPG image (max 10MB)
- **response_type** (required): Choose between `json` or `image`

### Response Modes

#### 1. JSON Response (`response_type=json`)
Returns detection data in JSON format:

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

#### 2. Image Response (`response_type=image`)
Returns annotated PNG image with:
- Bounding boxes around detected objects
- Labels with confidence scores
- Color-coded detection classes
- Detection count in response headers


![12](https://github.com/user-attachments/assets/d17d8988-72fc-4b8d-a254-d16ece3359da)
![17](https://github.com/user-attachments/assets/f63c5263-5cdf-4f52-b0c0-4f86bc07ffff)
![22](https://github.com/user-attachments/assets/362db0f0-cac8-451e-a54f-462e6bbb2c88)

### Usage Examples

#### cURL - JSON Response
```bash
curl -X POST "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict" \
     -F "file=@your_blueprint.png" \
     -F "response_type=json"
```

#### cURL - Image Response
```bash
curl -X POST "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict" \
     -F "file=@your_blueprint.png" \
     -F "response_type=image" \
     --output detected_result.png
```

#### Python - JSON Response
```python
import requests

url = "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict"
files = {"file": open("blueprint.png", "rb")}
data = {"response_type": "json"}

response = requests.post(url, files=files, data=data)
detections = response.json()["detections"]
print(f"Found {len(detections)} objects")
```

#### Python - Image Response
```python
import requests

url = "https://huggingface.co/spaces/kurakula-Prashanth2004/door-window-detection/predict"
files = {"file": open("blueprint.png", "rb")}
data = {"response_type": "image"}

response = requests.post(url, files=files, data=data)
with open("annotated_result.png", "wb") as f:
    f.write(response.content)
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

- **GPU Acceleration**: Automatically uses CUDA if available with FP16 precision
- **Model Warmup**: Dummy inference on startup for faster first request
- **Async Processing**: Non-blocking image processing with ThreadPoolExecutor (2 workers)
- **Smart Resizing**: Large images automatically resized to max 1280px
- **Memory Efficient**: Optimized for production deployment
- **Confidence Thresholding**: Filters low-confidence detections (≥0.5)
- **IoU Filtering**: Non-maximum suppression with 0.45 threshold
- **Color Consistency**: Hash-based color generation for detection labels

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

## 🔍 Model Configuration

- **Architecture**: YOLOv8 Medium (yolov8m_custom.pt)
- **Input Processing**: Auto-resize to max 1280px, maintains aspect ratio
- **Inference Settings**:
  - Confidence Threshold: 0.5
  - IoU Threshold: 0.45
  - Max Detections: 100
  - Half Precision: Enabled on GPU
- **Classes**: 2 (door, window)
- **Training Data**: Custom-labeled blueprint images

## 🎨 Visual Features

- **Dynamic Bounding Boxes**: Color-coded by detection class
- **Confidence Labels**: Shows class name and confidence score
- **Hash-based Colors**: Consistent colors for each label type
- **High-Quality Output**: PNG format with preserved image quality

## 🔧 API Configuration

- **File Size Limit**: 10MB maximum
- **Supported Formats**: JPG, PNG
- **Concurrent Processing**: 2 worker threads
- **Response Headers**: Include detection count metadata
- **Error Handling**: Comprehensive validation and error messages

## 📈 Results & Screenshots

### Training Progress
- Loss curves and training metrics
- Model performance on validation set
- Convergence after 100 epochs

### API Responses
- JSON detection data examples
- Annotated image outputs
- Performance benchmarks

### Interactive Documentation
- Swagger UI at `/docs`
- Parameter descriptions
- Live API testing interface

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
