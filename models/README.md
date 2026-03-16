---
license: other
tags:
- object-detection
- computer-vision
- yolo
- id-card
- document-ai
- pytorch
- onnx
library_name: ultralytics
pipeline_tag: object-detection
---

# ID Card Object Detection Model (YOLO)

## 📌 Overview

This model is a YOLO-based object detection model trained to detect regions on ID card images such as:

- ID card boundaries
- Face / portrait region
- Barcodes / QR codes
- Text regions / structured fields (depending on dataset)

The model is intended for:
- Document processing pipelines
- OCR preprocessing and region extraction
- Identity verification workflows
- Computer vision research and prototyping

The model outputs bounding boxes, class labels, and confidence scores.

---

## 🚀 How to Use

### Python (Ultralytics)

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('image.jpg', conf=0.25)
results[0].show()
```

### Download from Hugging Face

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id='miguelescamilla/id-card-yolo', filename='best.pt')
```

---

## 🧠 Model Details

- Architecture: YOLO (Ultralytics)
- Framework: PyTorch
- Input size: 640×640 (default)
- Task: Object Detection
- Outputs:
  - Bounding boxes (xyxy)
  - Class IDs
  - Confidence scores

---

## ⚠️ Limitations

- Performance depends on image quality, lighting, and camera perspective.
- Accuracy is limited by the size and diversity of the training dataset.
- Not validated for safety-critical or regulated environments.

---

## 📜 License & Credits

### Model Weights
This repository contains trained model weights uploaded by the author.

### YOLO Framework Credit
This model was trained using **Ultralytics YOLO**, licensed under the **AGPL-3.0 license**.

**Important:** If you use this model in commercial or proprietary systems, you must comply with Ultralytics licensing terms or obtain a commercial license.

Ultralytics Links:
- https://github.com/ultralytics/ultralytics
- https://www.ultralytics.com

---

## 👤 Author

Uploaded by: miguelescamilla
Last updated: 2026-01-12
