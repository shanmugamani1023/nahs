# Day 14 — DINOv2

## Concepts Covered
- Self-supervised learning
- Student-teacher framework
- Global + local views
- EMA teacher update
- DINOv2 vs CLIP
- Segmentation with DINOv2
- Depth estimation with DINOv2

---

## 1. Problem First

Every model so far needed labeled data:
- CLIP: image + text label pairs
- YOLO: bbox + class labels
- U-Net: pixel masks

**Labels are expensive.** DINOv2 learns powerful features from images with **zero labels.**

---

## 2. Core Idea — Self-supervised Learning

No human labels. The image itself is the supervision.

**Analogy:** You learn to understand cities by walking through them — no teacher, no textbook. Just observation.

---

## 3. Student-Teacher Framework

Two identical networks:

```
Teacher network (slowly updated)
Student network (learns from teacher)
```

```
Same image → two different augmented views

View 1 (global, 224×224) → Teacher → teacher_embedding
View 2 (local,  96×96)   → Student → student_embedding

Loss: student_embedding should match teacher_embedding
```

Student learns: "even from a small crop, I should understand the same thing as teacher."

**Analogy:** Senior doctor (teacher) examines full X-ray. Junior (student) sees only a crop. Junior must predict what senior would say. After thousands of X-rays — junior becomes as good as senior, without labels.

---

## 4. What Global + Local Training Forces

```
Training:
  Global view → Teacher → "this is a cat, sitting, on sofa"
  Local view  → Student → must predict same from small patch

After 1000s of images:
  Student learned: even from small patch of fur →
                   knows it's a cat, where it is, what surrounds it
```

Every patch understands:
- What object it belongs to
- Where it is in the scene
- What surrounds it

---

## 5. Teacher Update — EMA

Teacher NOT trained by backprop directly. Updated by Exponential Moving Average:

```
teacher_weights = 0.996 × teacher_weights + 0.004 × student_weights
```

Teacher = slow-moving average of student → stable training target.

**Analogy:** Teacher's knowledge = running average of everything student learned. Always slightly ahead, giving stable targets.

---

## 6. DINOv2 Improvements Over DINO v1

```
DINO v1 (2021):  ViT backbone, self-supervised
DINOv2 (2023):   + 142M curated images (larger dataset)
                 + ViT-g (1B parameters, larger model)
                 + better for dense tasks (segmentation, depth)
```

---

## 7. Linear Probe Results

Features so rich — just freeze DINOv2 + train one linear layer on top:

```
Task                     Accuracy
ImageNet classification  86.5%
Depth estimation         state of the art
Segmentation             state of the art
Instance retrieval       state of the art
```

**Analogy:** Perfect map — navigation is trivial. Rich features — any task is easy on top.

---

## 8. DINOv2 vs CLIP

| | CLIP | DINOv2 |
|--|------|--------|
| Supervision | Text-image pairs | No labels (self-supervised) |
| Strength | Image-text alignment | Pure visual features |
| Zero-shot classification | Yes (via text) | No (no text encoder) |
| Dense tasks (segmentation, depth) | Weak | Very strong |
| Feature quality | Good | Better for vision-only |

**Use CLIP when:** image+text alignment needed.
**Use DINOv2 when:** best pure visual features needed.

---

## 9. Segmentation with DINOv2

### Way 1: Attention Maps as Free Mask (No Training)

```
Image → DINOv2 → attention map
Attention map: high on cat pixels, low on background
Threshold → binary mask
```

```
Attention map:         Mask (threshold 0.5):
0.1  0.1  0.9  0.9     0  0  1  1
0.1  0.9  0.9  0.1  →  0  1  1  0
0.1  0.9  0.9  0.1     0  1  1  0
0.1  0.1  0.1  0.1     0  0  0  0
```

Quality: rough. Effort: zero.

**Analogy:** Ask DINOv2 "where were you looking?" → it points to cat → use that as mask.

---

### Way 2: DINOv2 + Lightweight Decoder

```
Image → DINOv2 (frozen) → rich features
                               ↓
                     Small conv decoder (trainable)
                               ↓
                          pixel mask
```

Quality: very good. Effort: train only decoder with few labels.

---

### Way 3: DINOv2 + SAM Combined

```
Image → DINOv2 → "object at region (x,y)"
                        ↓
              Use as prompt for SAM
                        ↓
              SAM → pixel-perfect mask
```

Quality: best of both. Effort: no training.

---

### When to Use Which

```
No labels available         → DINOv2 attention maps
Few labels (50-100)         → DINOv2 + lightweight decoder
Need pixel-perfect          → DINOv2 + SAM
Fixed classes, many labels  → U-Net or YOLO Seg
```

---

## 10. Depth Estimation with DINOv2

### What is Depth Estimation?

```
Input:  normal 2D image
Output: depth map — every pixel gets a distance value

Pixel (100, 200) → 2.5 meters away
Pixel (50, 80)   → 8.0 meters away
```

### Why DINOv2 is Perfect for Depth

CNN sees local patches → misses global depth cues.
DINOv2 sees full global context → understands near/far relationships.

```
Depth cues DINOv2 learned:
1. Object size      → smaller = farther
2. Texture gradient → finer texture = farther
3. Occlusion        → object behind another = farther
4. Atmospheric haze → blurry = farther
5. Vertical position → higher in image = usually farther
```

**Analogy:** DINOv2 memorized millions of 3D scenes. Show 2D photo → already knows depth from experience.

### How It Works

```
Image
  ↓
DINOv2 (frozen) → patch embeddings (rich features)
  ↓
Lightweight depth decoder (small trainable network)
  ↓
Depth map (H×W) — one distance value per pixel
```

### What Features Encode

```
Patch near camera:
  embedding = [large object, sharp texture, occludes background]
  → decoder learns: this = close

Patch far from camera:
  embedding = [small object, blurry texture]
  → decoder learns: this = far
```

### Real Model — Depth Anything

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline("depth-estimation", model="LiheYoung/depth-anything-large-hf")
image = Image.open("street.jpg")
depth = pipe(image)["depth"]  # depth map
```

### DINOv2 vs CNN for Depth

| | CNN | DINOv2 |
|--|-----|--------|
| Context | Local only | Global scene |
| Labels needed | Large LiDAR dataset | Very few |
| Boundaries | Blurry | Sharp |
| Generalization | Limited | Strong |

---

## 11. Code — Feature Extraction

```python
import torch
from PIL import Image
import torchvision.transforms as T

# Load DINOv2
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval()

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

image = Image.open("cat.jpg")
x = transform(image).unsqueeze(0)

with torch.no_grad():
    features = model(x)  # (1, 768) — rich visual embedding
```

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Self-supervised | No labels, image is supervision | Learn by walking cities |
| Student-teacher | Student matches teacher from partial view | Junior predicts senior's diagnosis |
| EMA update | Teacher = slow average of student | Running average of learned knowledge |
| Global+local views | Forces full object understanding from patch | Full X-ray vs cropped section |
| Attention maps | Object boundaries emerge naturally | Point to what you were looking at |
| Depth estimation | Global cues → distance per pixel | Memorized 3D scenes → reads depth from 2D |

---

## One-line Summary

> DINOv2 = self-supervised ViT trained with student-teacher framework on global+local views → learns rich visual features with zero labels → used for segmentation (attention maps / decoder) and depth estimation (global scene cues → distance map).
