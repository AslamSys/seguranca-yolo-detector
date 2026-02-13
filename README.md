# 🎯 YOLO Detector

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Segurança (Jetson Orin Nano)](https://github.com/AslamSys/_system/blob/main/hardware/seguranca/README.md)** → **seguranca-yolo-detector**

### Containers Relacionados (seguranca)
- [seguranca-brain](https://github.com/AslamSys/seguranca-brain)
- [seguranca-camera-stream-manager](https://github.com/AslamSys/seguranca-camera-stream-manager)
- [seguranca-face-recognition](https://github.com/AslamSys/seguranca-face-recognition)
- [seguranca-event-analyzer](https://github.com/AslamSys/seguranca-event-analyzer)
- [seguranca-alert-manager](https://github.com/AslamSys/seguranca-alert-manager)
- [seguranca-video-recorder](https://github.com/AslamSys/seguranca-video-recorder)

---

**Container:** `yolo-detector`  
**Ecossistema:** Segurança  
**Hardware:** Jetson Orin Nano 8GB  
**Posição no Fluxo:** Detecção de objetos em tempo real

---

## 📋 Propósito

Detector YOLOv8n otimizado com TensorRT para identificar pessoas, veículos, pets e objetos suspeitos em 4 câmeras simultâneas a 30 FPS com tracking DeepSORT.

---

## 🎯 Responsabilidades

### Primárias
- ✅ Detectar objetos com YOLOv8n (80 classes COCO)
- ✅ Tracking multi-objeto (DeepSORT)
- ✅ Processar 4 câmeras @ 30 FPS
- ✅ Bounding boxes + IDs persistentes
- ✅ Filtrar classes relevantes (person, car, truck, motorcycle, dog, cat)

### Secundárias
- ✅ Contagem de objetos por zona
- ✅ Detecção de linha cruzada
- ✅ Heatmaps de movimento
- ✅ Alertas de objetos abandonados

---

## 🔧 Tecnologias

### Core
- **YOLOv8n** - Ultralytics (nano model)
- **TensorRT** - FP16 precision
- **DeepSORT** - Multi-object tracking
- **OpenCV CUDA** - GPU-accelerated image processing

---

## 📊 Especificações Técnicas

### Performance
```yaml
Model: YOLOv8n (3.2M params)
Precision: FP16 (TensorRT)
Input: 640x640 (resized from 1080p)
Inference: 8-12 ms per frame
Throughput: 120 FPS (4 cameras @ 30 FPS each)
GPU Usage: 1.5 GB VRAM
CPU Usage: 50%
```

### Detecções
```yaml
Classes: person, car, truck, bus, motorcycle, bicycle, dog, cat, backpack, handbag, suitcase
Confidence Threshold: 0.5
NMS Threshold: 0.4
Max Detections: 100 per frame
```

---

## 🔌 Interfaces de Comunicação

### Input (NATS Subscribe)
```javascript
Topic: "seguranca.camera.frame"
```

### Output (NATS Publish)
```javascript
Topic: "seguranca.yolo.detections"
Payload: {
  "camera_id": "cam_1",
  "timestamp": 1732723200.123,
  "frame_number": 12345,
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [100, 200, 300, 600],  // [x1, y1, x2, y2]
      "tracking_id": "track_001",
      "center": [200, 400]
    },
    {
      "class": "car",
      "confidence": 0.88,
      "bbox": [500, 300, 900, 700],
      "tracking_id": "track_002"
    }
  ],
  "total_persons": 1,
  "total_vehicles": 1
}
```

---

## 🚀 Docker Compose

```yaml
yolo-detector:
  build:
    context: ./yolo-detector
    dockerfile: Dockerfile.tensorrt
  
  container_name: yolo-detector
  restart: unless-stopped
  
  runtime: nvidia
  
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - YOLO_MODEL=/models/yolov8n.engine
    - CONFIDENCE_THRESHOLD=0.5
    - NMS_THRESHOLD=0.4
    - NATS_URL=nats://mordomo-nats:4222
  
  volumes:
    - ./models:/models
    - /dev/video0:/dev/video0
  
  networks:
    - seguranca-net
    - shared-nats
  
  deploy:
    resources:
      limits:
        memory: 2G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## 🧪 Código Python

```python
import tensorrt as trt
import pycuda.driver as cuda
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from nats.aio.client import Client as NATS

# Load YOLOv8n with TensorRT

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Segurança (Jetson Orin Nano)](https://github.com/AslamSys/_system/blob/main/hardware/seguranca/README.md)** → **seguranca-yolo-detector**

### Containers Relacionados (seguranca)
- [seguranca-brain](https://github.com/AslamSys/seguranca-brain)
- [seguranca-camera-stream-manager](https://github.com/AslamSys/seguranca-camera-stream-manager)
- [seguranca-face-recognition](https://github.com/AslamSys/seguranca-face-recognition)
- [seguranca-event-analyzer](https://github.com/AslamSys/seguranca-event-analyzer)
- [seguranca-alert-manager](https://github.com/AslamSys/seguranca-alert-manager)
- [seguranca-video-recorder](https://github.com/AslamSys/seguranca-video-recorder)

---
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)  # FP16

# DeepSORT tracker

## 🔗 Navegação

**[🏠 AslamSys](https://github.com/AslamSys)** → **[📚 _system](https://github.com/AslamSys/_system)** → **[📂 Segurança (Jetson Orin Nano)](https://github.com/AslamSys/_system/blob/main/hardware/seguranca/README.md)** → **seguranca-yolo-detector**

### Containers Relacionados (seguranca)
- [seguranca-brain](https://github.com/AslamSys/seguranca-brain)
- [seguranca-camera-stream-manager](https://github.com/AslamSys/seguranca-camera-stream-manager)
- [seguranca-face-recognition](https://github.com/AslamSys/seguranca-face-recognition)
- [seguranca-event-analyzer](https://github.com/AslamSys/seguranca-event-analyzer)
- [seguranca-alert-manager](https://github.com/AslamSys/seguranca-alert-manager)
- [seguranca-video-recorder](https://github.com/AslamSys/seguranca-video-recorder)

---
tracker = DeepSort(max_age=30, n_init=3)

async def process_frame(frame, camera_id):
    # YOLO inference (TensorRT)
    results = model.predict(frame, conf=0.5, verbose=False)
    
    # Extract detections
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    
    # DeepSORT tracking
    detections = []
    for box, cls, conf in zip(boxes, classes, confidences):
        detections.append(([box[0], box[1], box[2]-box[0], box[3]-box[1]], conf, int(cls)))
    
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Publish to NATS
    payload = {
        "camera_id": camera_id,
        "timestamp": time.time(),
        "detections": [
            {
                "class": model.names[int(track.det_class)],
                "confidence": track.det_conf,
                "bbox": track.to_ltrb().tolist(),
                "tracking_id": f"track_{track.track_id:03d}"
            }
            for track in tracks if track.is_confirmed()
        ]
    }
    
    await nc.publish("seguranca.yolo.detections", json.dumps(payload).encode())
```

---

## 📚 Referências

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [DeepSORT](https://github.com/nwojke/deep_sort)

---

## 🔄 Changelog

### v1.0.0 (2024-11-27)
- ✅ YOLOv8n com TensorRT FP16
- ✅ DeepSORT tracking
- ✅ 4 câmeras @ 30 FPS
