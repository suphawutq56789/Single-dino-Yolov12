# YOLOv12 Triple Greyscale with DINOv3 Integration

YOLOv12 object detection model with **triple greyscale input** (3 images × 1 channel = 3 channels total) and optional DINOv3 feature extraction for enhanced performance.

## Features

- **Triple Greyscale Input**: Process 3 consecutive greyscale frames simultaneously (3 channels total)
- **DINOv3 Integration**: Optional DINOv3 backbone for improved feature extraction
- **Flexible Architecture**: Multiple integration strategies (P0 preprocessing, P3 enhancement, dual integration)
- **Memory Efficient**: Lower memory usage compared to RGB approaches
- **Multi-Scale Support**: 640px (YOLO standard, default) or 224px (DINOv3 optimal)

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Triple-dino-yolov12.git
cd Triple-dino-yolov12
pip install -e .
pip install transformers torch torchvision
```

### Dataset Preparation

Your dataset should follow YOLO format with greyscale images:

```
dataset/
├── images/
│   ├── primary/
│   │   ├── train/  # Greyscale images
│   │   └── val/
│   ├── left/       # (optional for triple input)
│   └── right/      # (optional for triple input)
└── labels/
    └── primary/
        ├── train/  # YOLO format labels (.txt)
        └── val/
```

**data.yaml** example:
```yaml
path: /path/to/dataset
train: images/primary/train
val: images/primary/val

nc: 1  # number of classes
names: ['object']

ch: 3  # 3-channel greyscale input
```

## Training Commands

### Basic Training (No DINOv3) - Fastest

```bash
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate nodino \
  --variant s \
  --batch 16 \
  --epochs 100 \
  --imgsz 640
```

### DINOv3 P0 Preprocessing (Recommended)

```bash
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate initial \
  --dinov3-size small \
  --variant s \
  --batch 16 \
  --epochs 100 \
  --imgsz 640
```

### DINOv3 P3 Feature Enhancement

```bash
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate p3 \
  --dinov3-size small \
  --variant m \
  --batch 8 \
  --epochs 100 \
  --imgsz 640
```

### Dual DINOv3 Integration (P0 + P3) - Maximum Accuracy

```bash
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate p0p3 \
  --dinov3-size small \
  --variant l \
  --batch 4 \
  --epochs 100 \
  --imgsz 640
```

## Model Variants

| Variant | Width Scale | Parameters | Use Case |
|---------|-------------|------------|----------|
| **n** | 0.25 | ~2.5M | Mobile/Edge devices, fastest |
| **s** | 0.50 | ~9M | Balanced speed/accuracy (default) |
| **m** | 1.00 | ~20M | High accuracy |
| **l** | 1.00 | ~26M | Production |
| **x** | 1.50 | ~59M | Maximum accuracy |

## Integration Strategies

### 1. `nodino` - Standard YOLO (Fastest)
No DINOv3 integration. Best for baseline or limited GPU memory.

```bash
--integrate nodino --batch 32
```

### 2. `initial` - P0 Preprocessing (Recommended)
DINOv3 preprocesses input before YOLO backbone. Best balance of accuracy and speed.

```bash
--integrate initial --dinov3-size small --batch 16
```

### 3. `p3` - P3 Feature Enhancement
DINOv3 enhances features after P3 stage. Good for fine-grained detection.

```bash
--integrate p3 --dinov3-size small --batch 8
```

### 4. `p0p3` - Dual Integration (Maximum Accuracy)
DINOv3 at both P0 and P3. Slower but most accurate.

```bash
--integrate p0p3 --dinov3-size small --batch 4
```

## Training Options

### Image Size

```bash
# 640px (YOLO standard, recommended)
--imgsz 640

# 224px (DINOv3 optimal, faster)
--imgsz 224

# 1024px (high resolution, slower)
--imgsz 1024
```

### Batch Size Recommendations

```bash
# Small GPU (4-6GB)
--batch 4 --variant n

# Medium GPU (8-12GB)
--batch 16 --variant s

# Large GPU (24GB+)
--batch 32 --variant l
```

### DINOv3 Model Sizes

**Note**: Use DINOv2 models (no authentication required)

| Size | Parameters | Speed | Accuracy |
|------|------------|-------|----------|
| **small** | 22M | ⚡⚡⚡ | Good (default) |
| **base** | 86M | ⚡⚡ | Better |
| **large** | 300M | ⚡ | Best |

```bash
# Small (recommended)
--dinov3-size small

# Base (higher accuracy)
--dinov3-size base

# Large (maximum accuracy)
--dinov3-size large
```

## Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/train/yolov12_triple_grey/weights/best.pt')

# Predict
results = model.predict(
    source='path/to/greyscale/images',
    conf=0.25,
    imgsz=640,
    save=True
)

# Process results
for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} objects")
```

## Validation

```bash
python val.py \
  --model runs/train/yolov12_triple_grey/weights/best.pt \
  --data dataset.yaml \
  --imgsz 640
```

## Export Models

```python
from ultralytics import YOLO

model = YOLO('runs/train/yolov12_triple_grey/weights/best.pt')

# Export to ONNX
model.export(format="onnx")

# Export to TensorRT
model.export(format="engine", half=True)
```

## Complete Training Examples

### Quick Start - Balanced Performance

```bash
# Recommended starting point
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate initial \
  --dinov3-size small \
  --variant s \
  --batch 16 \
  --epochs 100 \
  --imgsz 640
```

### Fast Training - No DINOv3

```bash
# Fastest training, good baseline
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate nodino \
  --variant s \
  --batch 32 \
  --epochs 100 \
  --imgsz 640
```

### High Accuracy - Large Model

```bash
# Maximum accuracy
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate p0p3 \
  --dinov3-size base \
  --variant l \
  --batch 4 \
  --epochs 200 \
  --imgsz 640
```

### Small GPU - Memory Optimized

```bash
# For GPUs with 4-6GB VRAM
python "train_grey scale.py" \
  --data dataset.yaml \
  --integrate nodino \
  --variant n \
  --batch 8 \
  --epochs 100 \
  --imgsz 416
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch 4

# Reduce image size
--imgsz 416

# Use smaller variant
--variant n

# Disable DINOv3
--integrate nodino
```

### Slow Training
```bash
# Use nodino (no DINOv3)
--integrate nodino --batch 32

# Reduce image size
--imgsz 224

# Use nano variant
--variant n
```

### DINOv3 Authentication Error
DINOv2 models are now used by default and don't require authentication. If you see auth errors, the code will automatically fall back to DINOv2.

## Performance Tips

1. **Start with `--integrate initial --variant s`** for best balance
2. **Use `--imgsz 640`** for standard YOLO performance
3. **Use `--integrate nodino`** for fastest training
4. **Adjust batch size** based on GPU memory
5. **Use variant `n`** for edge devices or real-time inference

## Command Line Arguments

```bash
# Required
--data YAML_FILE          # Dataset configuration

# Model Selection
--variant {n,s,m,l,x}     # Model size (default: s)
--integrate STRATEGY      # initial, nodino, p3, p0p3 (default: initial)
--dinov3-size SIZE        # small, base, large (default: small)

# Training Parameters
--epochs INT              # Training epochs (default: 100)
--batch INT               # Batch size (default: 8)
--imgsz INT               # Image size (default: 640)
--device DEVICE           # GPU device: 0, cpu, or 0,1,2,3

# Advanced Options
--patience INT            # Early stopping patience (default: 50)
--freeze-dinov3          # Freeze DINOv3 weights (default)
--unfreeze-dinov3        # Fine-tune DINOv3
--name NAME               # Experiment name
```

## Model Configuration Files

All configs support 3-channel greyscale input:

- `yolov12_triple_greyscale.yaml` - Base greyscale model
- `yolov12_triple_dinov3_p0_only.yaml` - P0 preprocessing only
- `yolov12_triple_dinov3_p3.yaml` - P3 enhancement only
- `yolov12_triple_dinov3_p0p3_adapter.yaml` - Dual integration

## License

AGPL-3.0 License

## Citation

```bibtex
@misc{yolov12-triple-greyscale,
  title={YOLOv12 Triple Greyscale with DINOv3 Integration},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/Triple-dino-yolov12}
}
```

## Acknowledgements

- YOLOv12 architecture by Yunjie Tian et al.
- DINOv2 by Meta AI
- Ultralytics YOLO implementation
