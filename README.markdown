# YOLOv5: Detecting Chess Pieces

## Overview

This project demonstrates how to train and deploy a YOLOv5 model for detecting and classifying chess pieces (black and white bishops, kings, knights, pawns, queens, and rooks) on a chessboard. The workflow leverages a publicly available dataset from Roboflow and Google Colab for training, testing, and evaluating the model.

The direct Colab notebook link for this project is not publicly available, but the code and resources are provided below.

### Resources:
- **Notebook:** Chess_Piece_Detection_YOLOv5.ipynb
- **Model Weights:** best.pt
- **Test Video:** Chess Piece Detection.mp4
- **Test Output:** output_chess.mp4
- **Dataset Source:** [Roboflow Dataset](https://universe.roboflow.com/cifar10-image-classification/chess-piece-detection-omx7i)

## Dataset Information

The dataset, sourced from Roboflow, contains images of chessboards with annotated chess pieces. It is organized into training, validation, and test sets, with annotations for 12 classes representing black and white chess pieces.

### Dataset Summary:
- **Number of Images:** Varies (specific count available in Roboflow project)
- **Classes (12):**
  - black_bishop, black_king, black_knight, black_pawn, black_queen, black_rook
  - white_bishop, white_king, white_knight, white_pawn, white_queen, white_rook

---
- **Dataset Link:** [Chess Piece Detection Dataset](https://universe.roboflow.com/cifar10-image-classification/chess-piece-detection-omx7i)

# Step-by-Step Workflow

## Step 1: Setting Up the Environment

Verify GPU availability, mount Google Drive to access or save resources, and ensure the environment is ready.

```python
# Import necessary libraries and check for GPU
import torch
from google.colab import drive

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Mount Google Drive
drive.mount('/content/drive')
print("Drive mounted. Your dataset will be accessible at /content/drive/MyDrive/ChessPieceDetection")
```

---

## Step 2: Cloning YOLOv5 and Installing Dependencies

Clone the YOLOv5 repository and install required dependencies, including Roboflow for dataset access.

```bash
!pip uninstall ultralytics -y
%cd /content
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt -q
!pip install roboflow -q
```

---

## Step 3: Downloading and Previewing the Dataset

Download the chess piece detection dataset from Roboflow and inspect its structure.

```python
from roboflow import Roboflow

# Download dataset
rf = Roboflow(api_key="THWAfydGbO78ypAyYu7H")
project = rf.workspace("cifar10-image-classification").project("chess-piece-detection-omx7i")
version = project.version(1)
dataset = version.download("yolov5")

# List dataset contents
dataset_path = dataset.location
!ls "{dataset_path}"
!ls "{dataset_path}/train/images" | head -n 5
!ls "{dataset_path}/valid/images" | head -n 5
!ls "{dataset_path}/test/images" | head -n 5
```

---

## Step 4: Preparing the Configuration File

Create a `data.yaml` file to specify dataset paths, number of classes, and class names.

```python
import glob
import os

# Get unique classes from labels
def get_unique_classes(label_dir):
    class_ids = set()
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_ids.add(class_id)
    return sorted(list(class_ids))

train_label_dir = os.path.join(dataset_path, "train/labels")
unique_classes = get_unique_classes(train_label_dir)
class_names = [
    'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
    'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
]
num_classes = len(unique_classes)

# Create data.yaml
yaml_content = f"""train: {dataset_path}/train/images
val: {dataset_path}/valid/images
nc: {num_classes}
names: {class_names}"""

yaml_path = f"{dataset_path}/data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
```

---

## Step 5: Training the YOLOv5 Model

Train the YOLOv5 model using the chess piece dataset with specified hyperparameters.

```bash
!python train.py \
  --data {dataset_path}/data.yaml \
  --cfg yolov5s.yaml \
  --weights yolov5s.pt \
  --epochs 50 \
  --batch 16 \
  --img 640 \
  --name chess_piece_exp \
  --cache
```

---

## Step 6: Saving Model Weights

Save the trained model weights to Google Drive for later use.

```bash
!ls runs/train/chess_piece_exp/weights/
!zip -r best.zip runs/train/chess_piece_exp/weights/best.pt
!mkdir -p /content/drive/MyDrive/ChessPieceDetection
!cp runs/train/chess_piece_exp/weights/best.pt /content/drive/MyDrive/ChessPieceDetection/
```

---

## Step 7: Testing the Model on a Video

Test the trained model on a video to detect chess pieces. The output is an annotated video with bounding boxes.

```python
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.plots import Annotator, colors
import torch
import numpy as np

# Ensure OpenCV is installed
!pip install opencv-python -q

# Copy best.pt to YOLOv5 directory
!cp /content/drive/MyDrive/ChessPieceDetection/best.pt /content/yolov5/

# Load YOLOv5 model
weights_path = '/content/yolov5/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights_path, device=device)

# Video file path
video_path = '/content/drive/MyDrive/Aivancity/Deep Learning/Chess Piece Detection.mp4'

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video at '{video_path}'.")
    exit()

# Video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_chess.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for YOLO
    img = frame[:, :, ::-1].copy()  # Convert BGR to RGB
    img = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension

    # Run inference
    results = model(img)
    detections = non_max_suppression(results)

    # Annotate frame
    annotator = Annotator(frame, line_width=2, example="YOLOv5")
    for det in detections:
        if det is not None and len(det):
            for *box, conf, cls in det:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                annotator.box_label(box, label, color=colors(int(cls), True))

    annotated_frame = annotator.result()
    out.write(annotated_frame)

cap.release()
out.release()
print("Output video saved as 'output_chess.mp4'")

# Download the output video
from google.colab import files
files.download('output_chess.mp4')
```

---

## Results

### Test Outputs
- **Test Images:** Annotated detection results are saved in the `runs/detect/chess_image_exp` folder (if image inference is run).
- **Test Video Output:** The processed video is saved as `output_chess.mp4`.

### Performance Insights
Training and validation metrics (e.g., precision, recall, mAP) are generated during training and can be visualized in the `runs/train/chess_piece_exp` folder (e.g., `results.png`). The model accurately detects and classifies chess pieces across various board configurations.

---

## Conclusion

This project provides a complete workflow for training and deploying YOLOv5 for chess piece detection. With its robust performance and fast inference, YOLOv5 is well-suited for real-time object detection tasks in game analysis and computer vision applications. Use the provided code and dataset to replicate or extend this project for your own datasets!
