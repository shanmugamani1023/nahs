# Day 9 — Object Detection

## Concepts Covered
- Sliding window
- Anchor boxes
- YOLO pipeline (full)
- Training with loss
- NMS, IoU
- Model comparison (R-CNN family vs YOLO)
- YOLO version history

---

## 1. Problem

```
Classification: whole image = "cat"
Detection:      WHERE is the cat? + WHAT is it? → bbox + class
```

---

## 2. Sliding Window (Old)

```
Image → crop window → classify → slide → repeat
```

**Problem:** Too slow. Millions of crops × multiple scales.

**Analogy:** Security guard checks every single locker individually.

---

## 3. Grid + Anchor Boxes

CNN naturally compresses 416×416 → 13×13 feature map.
Each cell = 32×32 pixel region of original image.

Each cell is responsible for objects whose **center falls in that cell**.

Each cell predicts **3 anchor boxes** (pre-defined shapes from k-means on training data).

**Analogy:** Parking lot with zones. Each zone has 3 marked spots (bike, car, truck).

---

## 4. What Each Anchor Predicts

```
[tx, ty, tw, th, confidence, class1, class2, ...]

tx, ty      = center offset within cell (0 to 1)
tw, th      = scale relative to anchor size
confidence  = P(object) × IoU
class probs = probability per class
```

For 80 COCO classes:
```
1 anchor = 5 + 80 = 85 numbers
3 anchors per cell = 255
Total output = 13 × 13 × 255
```

---

## 5. Full YOLO Pipeline

```
416×416 image
    ↓
CNN backbone (learns features)
    ↓
13×13×512 feature map
    ↓
Prediction head (1×1 conv)
    ↓
13×13×255 output tensor
    ↓
Decode boxes → filter by confidence > threshold
    ↓
NMS → remove duplicates
    ↓
Final detections
```

**Analogy:** One camera sees entire parking lot at once → instantly reports all vehicles.

---

## 6. Small Number Example

8×8 image → 2×2 grid, 1 anchor, 2 classes:

```
Each cell output: [tx, ty, tw, th, conf, P(cat), P(dog)]  = 7 numbers
Total: 2×2×7 = 28 numbers

Cell A (has cat):  [0.5, 0.5, 1.2, 0.8, 0.95, 0.90, 0.05]
Cell B (empty):    [0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.01]
```

Decode cell A box:
```
actual_cx = (cell_x + tx) × cell_size = (0 + 0.5) × 4 = 2
actual_cy = (cell_y + ty) × cell_size = (0 + 0.5) × 4 = 2
actual_w  = anchor_w × e^tw = 4 × e^1.2 = 13.2
```

---

## 7. Training — 3 Loss Cases

```
Case 1: Cell HAS object, wrong prediction
  → bbox loss + confidence loss + class loss

Case 2: Cell EMPTY, model predicted high confidence (hallucination)
  → confidence loss (push conf toward 0)

Case 3: Multiple anchors — only best IoU anchor gets trained for object
  → others treated as empty → conf loss only
```

**How confidence changes via backprop:**
```
conf_pred = sigmoid(raw_value)

Loss = (conf_pred - conf_target)²
Gradient = 2 × (conf_pred - conf_target)

raw_new = raw_old - lr × gradient

If conf too high → gradient positive → raw decreases → conf drops
If conf too low  → gradient negative → raw increases → conf rises
```

**Analogy:** Student raises hand in empty zone → teacher deducts 1 mark → after 1000 deductions, student stops raising hand there.

---

## 8. IoU

```
IoU = Area of Overlap / Area of Union

IoU = 1.0 → perfect match
IoU > 0.5 → correct detection
IoU = 0.0 → no overlap
```

**Analogy:** Two people circle the same object. IoU = how well their circles agree.

---

## 9. NMS — Non-Maximum Suppression

Multiple boxes for same object → keep highest confidence, remove overlapping ones.

```
Box1: conf=0.9  ← keep
Box2: conf=0.7, IoU(Box1,Box2)=0.8 → overlaps → remove
Box3: conf=0.6, IoU(Box1,Box3)=0.7 → overlaps → remove
```

**Analogy:** Three students raise hand for same answer → teacher picks most confident one.

---

## 10. Model Comparison

```
Model          Speed (fps)    mAP
R-CNN          0.02           ~53
Fast R-CNN     0.5            ~70
Faster R-CNN   5-7            ~73
SSD            46             ~74
YOLOv3         30             ~82
YOLOv5         140            ~85
YOLOv8         160            ~87
YOLOv11        200+           ~90
```

**R-CNN family = two-stage** (propose regions → classify)
**YOLO = one-stage** (predict directly in one pass)

---

## 11. YOLO Version History

```
v1-v4:   Darknet (C framework) — fast but hard to use
v5:      First Pythonic YOLO (PyTorch, pip install) — industry adopted massively
v6, v7:  Incremental improvements
v8:      Anchor-free, supports detection+segmentation+pose+classification
v9, v10: Research improvements
v11:     Latest, best accuracy+speed
```

**Use:**
- Learning: YOLOv3
- Production: YOLOv8
- Edge devices: YOLOv8n (nano)

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Grid | CNN output naturally 13×13 | Room summary table |
| Anchor | Pre-defined box shapes per cell | Pre-marked parking spots |
| Confidence | sigmoid(raw) → 0-1 | Volume knob limited to 0-1 |
| NMS | Remove overlapping boxes, keep best | Teacher picks most confident student |
| IoU | Overlap/Union | How well two circles agree |
| One-stage | Predict bbox+class in one pass | One camera, instant report |
