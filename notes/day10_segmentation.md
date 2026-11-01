# Day 10 — Segmentation

## Concepts Covered
- Semantic vs Instance vs Panoptic segmentation
- U-Net architecture
- Encoder-Decoder structure
- Skip connections in U-Net
- Loss functions (cross-entropy, Dice)

---

## 1. Three Types

```
Classification:  whole image label        → "cat"
Detection:       bounding box + label     → box around cat
Segmentation:    every pixel labeled      → cat pixels vs background pixels
```

---

## 2. Semantic Segmentation

Every pixel gets a class label. Same class = same color, no instance distinction.

```
Input:           Output:
🐱 🐱 🌿 🌿     C  C  G  G
🐱 🐱 🌿 🌿     C  C  G  G    ← both cats = same color C
🌿 🌿 🌿 🌿     G  G  G  G
🐱 🐱 🌿 🌿     C  C  G  G
```

**Problem:** Two cats → same label, can't tell them apart.

**Analogy:** Paint by numbers. Cat = orange, grass = green. Two cats = same orange. Know WHAT, not HOW MANY.

---

## 3. Instance Segmentation

Each object instance gets unique label at pixel level.

```
Output:
Cat1 Cat1  Grass  Grass
Cat1 Cat1  Grass  Grass
Grass Grass Grass Grass
Cat2 Cat2  Grass  Grass   ← Cat1 ≠ Cat2
```

**Analogy:** Cat1 = orange, Cat2 = red. Know WHAT and HOW MANY.

---

## 4. Panoptic Segmentation

Semantic + Instance combined.

```
"Things" (countable: cats, cars) → instance level
"Stuff"  (uncountable: sky, grass) → semantic level
```

**Analogy:** Grass = all one green (semantic). Each cat = unique color (instance). Full scene understanding.

---

## 5. Why Plain CNN Fails for Segmentation

CNN downsamples: 512×512 → 16×16 (loses spatial info).

Segmentation needs pixel-level output → must go back to 512×512.

Solution: **Encoder-Decoder with Skip Connections = U-Net**

---

## 6. U-Net Architecture

```
Encoder (compress):              Decoder (expand):

512×512×3  ──────────────────────────────── 512×512×2 (mask)
    ↓                                ↑
256×256×64 ──────────────────── 256×256×64
    ↓                        ↑
128×128×128 ──────────── 128×128×128
    ↓                ↑
 64×64×256 ──────  64×64×256
    ↓        ↑
  32×32×512 (bottleneck)
```

Shape = U → U-Net

---

## 7. Encoder (Left Side)

Standard CNN — conv + maxpool → compress + extract features.

```
512×512×3
    ↓ Conv+ReLU
512×512×64   → SAVED for skip connection
    ↓ MaxPool
256×256×64
    ↓ Conv+ReLU
256×256×128  → SAVED for skip connection
    ↓ MaxPool
...
32×32×512    ← bottleneck (richest features, smallest spatial size)
```

---

## 8. Decoder (Right Side)

Upsample back to original size.

```
32×32×512
    ↓ Upsample (×2) → 64×64×512
    + concat skip connection from encoder (64×64×256)
    = 64×64×768
    ↓ Conv → 64×64×256
    ↓ Upsample (×2) → 128×128×256
...
512×512×2   ← 2 channels = 2 classes (cat, background)
```

---

## 9. Why Skip Connections?

Bottleneck = rich features but lost WHERE exactly.
Skip connections bring back spatial precision from encoder.

```
Encoder:    "cat-ear texture at pixel (120, 80)"
Bottleneck: "cat exists somewhere"
Skip+Decode: "cat at pixel (120, 80)" → precise pixel mask
```

**Analogy:** Drawing map from memory (blurry) + you kept photos from each zoom level. Combine → precise final map.

---

## 10. Encoder vs Decoder — WHAT vs WHERE

```
Encoder:          WHAT (features)       — loses WHERE
Skip connections: WHERE (coordinates)   — preserved from encoder
Decoder:          combines both         — pixel-level understanding
Output:           H×W mask             — every pixel labeled
```

### Encoder — WHAT
```
Image → Conv layers → Feature maps

Learns:
"There is fur texture here"
"There is a round shape here"
"There are pointy ears here"
```
Answers: WHAT features exist. But spatial precision lost due to downsampling (512→32).

### Bottleneck — WHAT (most abstract)
```
32×32×512 → "Cat-like pattern exists somewhere in this region"
```
Richest features, but location is very rough.

### Decoder — WHERE
```
Upsamples: 32 → 64 → 128 → 256 → 512
```
Answers: WHERE those features are. But decoder alone is blurry — spatial info was lost during encoding.

### Skip Connections — WHAT + WHERE together
```
Encoder layer:    "fur texture at pixel (120, 80)"  ← precise location
Bottleneck:       "cat exists"                       ← no precise location
Decoder + skip:   "cat at pixel (120, 80)"           ← both combined
```

**Analogy:** Encoder = detective collects clues (fur, shape, ears) but forgets exact room locations. Skip connection = photos taken at each room during investigation. Decoder = writes final report using clues + photos → "cat at pixel (120,80)."

### How Mask Value is Assigned (Argmax)

Network outputs probability per class per pixel:
```
Output tensor: H × W × num_classes

Pixel(1,1): [P(bg)=0.05, P(cat)=0.92, P(dog)=0.03] → argmax = 1 → cat
Pixel(0,0): [P(bg)=0.97, P(cat)=0.02, P(dog)=0.01] → argmax = 0 → background
Pixel(3,4): [P(bg)=0.04, P(cat)=0.03, P(dog)=0.93] → argmax = 2 → dog
```

```python
mask = np.argmax(output, axis=-1)
# mask[1,1] = 1  ← cat
# mask[0,0] = 0  ← background
```

Network learned "what cat pixels look like" from thousands of human-labeled masks during training.

---

## 11. Loss Functions

### Pixel-wise Cross Entropy
```
For each pixel:
  Loss = -log(P(correct_class))

Sum over all pixels = total loss
```

### Dice Loss (common in medical imaging)
```
Dice = 2 × |Prediction ∩ Ground Truth|
           ──────────────────────────
           |Prediction| + |Ground Truth|

Dice Loss = 1 - Dice

Dice=1.0 → perfect overlap → Loss=0
Dice=0.0 → no overlap → Loss=1
```

**Analogy:** Dice = how much two circles overlap. Minimize 1-overlap = maximize overlap.

---

## 12. YOLO Segmentation — Detection + Segmentation in One Model

### Architecture

```
Image
  ↓
CNN backbone
  ↓
Feature map
  ↓              ↓
Detection head   Mask head
  ↓              ↓
bbox+class+      32 prototype masks (whole image)
32 coefficients
  ↓
Linear combination: coefficients × prototypes
  ↓
Sigmoid → binary mask
  ↓
Crop to bbox
  ↓
Final: bbox + class + pixel mask per object
```

### Step 1: Prototype Masks

Model generates 32 prototype masks for the whole image.
These are building blocks — not final masks.

```
proto_1: responds to "left side objects"
proto_2: responds to "round shapes"
proto_3: responds to "fur textures"
...
```

**Analogy:** 32 stencils, each highlighting a different pattern.

### Step 2: Mask Coefficients Per Detection

Each detected object also outputs 32 coefficients:

```
Detected cat → [0.8, -0.3, 0.5, 0.1, ...]   32 numbers
Detected dog → [0.2,  0.7, -0.1, 0.9, ...]   32 numbers
```

**Analogy:** Recipe for mixing the 32 stencils. Different object = different recipe.

### Step 3: Linear Combination

```
cat_mask = coeff_1×proto_1 + coeff_2×proto_2 + ... + coeff_32×proto_32
```

→ sigmoid → binary mask → crop to bbox → final instance mask.

### Why Prototypes Instead of Direct Mask?

```
Direct:    80 objects × 640×640 = 32M values  → too slow
Prototype: 32 masks + 80×32 coefficients = 415K values → fast
```

**Analogy:** 32 base templates mixed differently per person. Same result, much faster.

### YOLO Seg vs U-Net

| | U-Net | YOLO Seg |
|--|-------|----------|
| Task | Semantic segmentation | Instance segmentation |
| Output | One mask for whole image | Separate mask per object |
| Speed | Slower | Real-time |
| Tells apart instances | No | Yes |
| Use case | Medical, satellite | Real-time object masking |

### Code (YOLOv8 Segmentation)

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n-seg.pt")  # n=nano, s=small, m=medium, l=large

# Run on image
results = model("image.jpg")

# Access results
for result in results:
    boxes = result.boxes        # bbox + class + confidence
    masks = result.masks        # pixel masks per object

    # Draw on image
    annotated = result.plot()
    cv2.imwrite("output.jpg", annotated)

# Access individual mask
mask = masks[0].data.cpu().numpy()   # H×W binary mask for first object
box  = boxes[0].xyxy.cpu().numpy()  # [x1, y1, x2, y2]
cls  = int(boxes[0].cls)            # class ID
conf = float(boxes[0].conf)         # confidence
```

### How Detection Head + Mask Head Work Together

```
Detection head  →  bbox + class + 32 coefficients (HOW TO MIX)
Mask head       →  32 prototype masks              (WHAT TO MIX)

Combine:
  32 coefficients × 32 prototypes → raw mask
  sigmoid → binary mask
  crop to bbox → final mask
```

```
                 ┌─── Detection head → bbox + [c1, c2, ... c32]
Feature map ────┤                                    ↓
                 └─── Mask head → [proto1, proto2, ... proto32]
                                                     ↓
                              c1×proto1 + c2×proto2 + ... = raw mask
                                                     ↓
                                                  sigmoid
                                                     ↓
                                              crop to bbox
                                                     ↓
                                           final instance mask
```

- **Mask head says:** "here are 32 patterns across the whole image"
- **Detection head says:** "cat is HERE, mix the 32 patterns like THIS"
- **Together:** "cat pixels = exactly these pixels"

Same 32 prototypes for all objects. Different coefficients per object → different final mask.

### One-line Summary

> YOLO Seg = normal YOLO + mask head that creates 32 prototype masks and mixes them differently per detected object → pixel-perfect instance mask at real-time speed.

---

## Quick Reference

| Type | Labels | Tells You |
|------|--------|-----------|
| Semantic | Per pixel, by class | What each pixel is |
| Instance | Per pixel, by object | What + which object |
| Panoptic | Both combined | Full scene understanding |

| U-Net Part | Role | Analogy |
|------------|------|---------|
| Encoder | Extract features, compress | Detective gathering clues |
| Bottleneck | Richest features, smallest size | Case summary |
| Decoder | Expand to original size | Drawing final map |
| Skip connections | Restore spatial precision | Photos from each zoom level |

| Loss | Use When |
|------|----------|
| Cross entropy | General segmentation |
| Dice loss | Class imbalance (medical: small lesion in big background) |
