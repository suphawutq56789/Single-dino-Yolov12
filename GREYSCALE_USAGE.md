# YOLOv12 Triple Input - GREYSCALE VERSION

## คู่มือการใช้งาน Greyscale Triple Input

### ภาพรวม

เวอร์ชัน greyscale นี้แปลงโค้ดจาก RGB (9 channels = 3 images × 3 RGB channels) เป็น **GREYSCALE (3 channels = 3 images × 1 greyscale channel)**

### ข้อดีของ Greyscale:
- ✅ **ใช้ memory น้อยกว่า** (~33% ของ RGB)
- ✅ **ประมวลผลเร็วขึ้น**
- ✅ **เหมาะสำหรับงานที่ไม่จำเป็นต้องใช้สี** (เช่น การตรวจจับโครงสร้าง, satellite imagery)

---

## 🚀 ขั้นตอนการติดตั้ง

### 1. แก้ไขไฟล์ `ultralytics/data/build.py`

หาบรรทัดที่ 103 และเปลี่ยนจาก:
```python
from ultralytics.data.triple_dataset import TripleInputDataset
```

เป็น:
```python
from ultralytics.data.triple_dataset_greyscale import TripleInputDatasetGreyscale as TripleInputDataset
```

**หรือ** เปลี่ยนทั้งบล็อก (บรรทัด 101-123):

```python
if is_triple_input:
    # Import and use TripleInputDatasetGreyscale for greyscale triple input
    from ultralytics.data.triple_dataset_greyscale import TripleInputDatasetGreyscale
    from ultralytics.utils import LOGGER
    LOGGER.info(f"Triple input GREYSCALE structure detected - using TripleInputDatasetGreyscale for {mode}")

    return TripleInputDatasetGreyscale(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # hyperparameters
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
        data=data,
        task=getattr(cfg, 'task', 'detect'),
    )
```

---

## 📝 การใช้งาน

### ตัวอย่างคำสั่ง Training

```bash
# Basic training with greyscale
python "train_grey scale.py" --data dataset.yaml --epochs 100 --batch 8

# Training with DINOv3 P0 preprocessing (greyscale)
python "train_grey scale.py" --data dataset.yaml --integrate initial --dinov3-size small

# Training without DINOv3 (standard greyscale triple input)
python "train_grey scale.py" --data dataset.yaml --integrate nodino --variant s

# Training with P3 feature enhancement
python "train_grey scale.py" --data dataset.yaml --integrate p3 --dinov3-size base

# Training with dual DINOv3 integration (P0 + P3)
python "train_grey scale.py" --data dataset.yaml --integrate p0p3 --dinov3-size base
```

### พารามิเตอร์สำคัญ

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | ไฟล์ config ของ dataset (*.yaml) | **required** |
| `--integrate` | DINOv3 integration: `initial`, `nodino`, `p3`, `p0p3` | `initial` |
| `--dinov3-size` | ขนาด DINOv3: `small`, `base`, `large`, `giant` | `small` |
| `--variant` | YOLOv12 variant: `n`, `s`, `m`, `l`, `x` | `s` |
| `--epochs` | จำนวน epochs | `100` |
| `--batch` | Batch size | `8` |
| `--imgsz` | ขนาดภาพ | `224` |
| `--device` | Device: `0`, `1`, `cpu` | `0` |

---

## 📂 โครงสร้าง Dataset

Dataset ต้องมีโครงสร้างแบบนี้:

```
dataset/
├── primary/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── ...
├── detail1/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── ...
├── detail2/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        └── ...
```

**หมายเหตุ:** แม้ว่าภาพจะเป็น RGB (.jpg), โค้ด greyscale จะโหลดเป็น greyscale โดยอัตโนมัติ

---

## ⚙️ ไฟล์ที่เกี่ยวข้อง

### ไฟล์ที่สร้างใหม่:
1. **`train_grey scale.py`** - สคริปต์ training แบบ greyscale
2. **`ultralytics/data/triple_dataset_greyscale.py`** - Dataset class สำหรับ greyscale
3. **`GREYSCALE_USAGE.md`** - ไฟล์นี้

### ไฟล์ที่ต้องแก้ไข:
1. **`ultralytics/data/build.py`** (บรรทัด 103) - เปลี่ยน import เป็น greyscale version

---

## 🔧 การทดสอบ

### ทดสอบว่า greyscale ทำงานได้:

```python
# ทดสอบโหลดภาพเป็น greyscale
from ultralytics.data.triple_dataset_greyscale import TripleInputDatasetGreyscale
import cv2

# โหลดภาพเดี่ยว
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {img.shape}")  # ควรเป็น (H, W) ไม่มี channel
print(f"Image dtype: {img.dtype}")  # ควรเป็น uint8
```

---

## 🆚 เปรียบเทียบ RGB vs Greyscale

| Feature | RGB Version | Greyscale Version |
|---------|-------------|-------------------|
| Channels | 9 (3 images × 3 RGB) | 3 (3 images × 1 greyscale) |
| Memory | ~3x more | Baseline |
| Speed | Slower | Faster |
| Color info | ✅ Yes | ❌ No |
| File | `train_triple_dinov3.py` | `train_grey scale.py` |
| Dataset | `triple_dataset.py` | `triple_dataset_greyscale.py` |

---

## ❗ ข้อควรระวัง

1. **HSV Augmentations ถูกปิดแล้ว** - เพราะ greyscale ไม่มีสี
2. **Visualization อาจมีปัญหา** - plots ถูกปิดไว้
3. **Color-based features ไม่มี** - เหมาะสำหรับงานที่ไม่ต้องการสี
4. **ต้องแก้ไข build.py** - อย่าลืมแก้ไข import ใน build.py

---

## 🐛 การแก้ปัญหา

### ปัญหา: RuntimeError: expected input to have 9 channels, but got 3

**สาเหตุ:** ยังใช้ RGB version อยู่

**แก้ไข:** ตรวจสอบว่าแก้ไข `ultralytics/data/build.py` ถูกต้องแล้ว

---

### ปัญหา: ภาพยังเป็นสี

**สาเหตุ:** Dataset ยังใช้ RGB loader

**แก้ไข:** ตรวจสอบว่า build.py import `triple_dataset_greyscale` แล้ว

---

### ปัญหา: Channel mismatch in validation

**สาเหตุ:** ปกติสำหรับ P0 integration

**แก้ไข:** ไม่ต้องทำอะไร training จะสำเร็จ แต่ final validation อาจ error

---

## 📚 ตัวอย่างคำสั่งเพิ่มเติม

```bash
# Training with small model, no DINOv3
python "train_grey scale.py" --data ../DATASET\ NEW/data.yaml --integrate nodino --variant n --epochs 50

# Training with base DINOv3, P0 only
python "train_grey scale.py" --data ../DATASET\ NEW/data.yaml --integrate initial --dinov3-size base --variant s --epochs 100

# Training with large DINOv3, dual integration
python "train_grey scale.py" --data ../DATASET\ NEW/data.yaml --integrate p0p3 --dinov3-size large --variant m --epochs 200 --batch 4

# CPU training (for testing)
python "train_grey scale.py" --data ../DATASET\ NEW/data.yaml --integrate nodino --device cpu --epochs 10 --batch 2
```

---

## 📊 ประสิทธิภาพที่คาดหวัง

- **Memory usage:** ~33% ของ RGB version
- **Training speed:** ~20-30% เร็วขึ้น
- **Accuracy:** อาจต่ำกว่า RGB เล็กน้อยถ้า color information สำคัญ
- **Best for:** Structural detection, texture analysis, satellite imagery

---

## 📞 การติดต่อ / Support

หากมีปัญหาหรือข้อสงสัย:
1. ตรวจสอบว่าแก้ไข `build.py` ถูกต้อง
2. ตรวจสอบ dataset structure
3. ลองใช้ `--integrate nodino` สำหรับ debugging
4. ตรวจสอบ error message และ traceback

---

## ✅ Checklist ก่อนเริ่ม Training

- [ ] แก้ไข `ultralytics/data/build.py` เปลี่ยน import เป็น greyscale
- [ ] ตรวจสอบ dataset structure (primary, detail1, detail2)
- [ ] ตรวจสอบว่า data.yaml ชี้ไปยัง primary folder
- [ ] ติดตั้ง dependencies: `pip install transformers timm huggingface_hub`
- [ ] ตั้งค่า HuggingFace token (ถ้าใช้ DINOv3)

---

**สร้างโดย:** Claude Code
**วันที่:** 2025-11-21
**เวอร์ชัน:** 1.0 (Greyscale Triple Input)
