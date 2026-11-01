# Day 13 — SAM (Segment Anything Model)

## Concepts Covered
- Problem with fixed-class segmentation
- Prompt-based segmentation
- Image encoder (ViT-H)
- Prompt encoder
- Mask decoder
- Zero-shot capability
- SAM vs YOLO Seg vs U-Net
- SAM 2 (video)

---

## 1. Problem First

Normal segmentation (U-Net, YOLO Seg):
- Trained on fixed classes
- New object → retrain

**SAM:** Segment ANY object, even ones never seen before.

```
Click on object → SAM segments it
Draw box around object → SAM segments it
No class labels needed
```

---

## 2. Core Idea — Prompt-based Segmentation

SAM doesn't ask "what is this?" — just "where is this?"

```
User gives prompt:
  Point (x, y)        → "segment whatever is at this point"
  Box (x1,y1,x2,y2)   → "segment whatever is inside this box"
  Text (optional)     → "segment the cat"

SAM → pixel mask of that object
```

**Analogy:** You point at something in a photo. SAM draws a perfect outline around it — without needing to know what it is.

---

## 3. Architecture — 3 Parts

```
Image  → [Image Encoder]  → image embedding
Prompt → [Prompt Encoder] → prompt embedding
              ↓
         [Mask Decoder] ← both embeddings combined
              ↓
         pixel mask
```

---

## 4. Part 1: Image Encoder (ViT-H)

Largest ViT model (ViT-Huge).

```
Image (1024×1024)
    ↓
ViT-H
    ↓
64×64×256 image embedding
```

Run **once per image** — expensive but done only once.
After this, any number of prompts applied cheaply.

**Analogy:** Photographer takes one high-quality scan of entire scene. After that, anyone can point to any object for free.

---

## 5. Part 2: Prompt Encoder

Converts user prompt into embedding.

```
Point prompt:  (x=200, y=150) + foreground/background label
               → positional encoding → 256-dim vector

Box prompt:    (x1,y1,x2,y2)
               → encode corners → 256-dim vector

Text prompt:   "the cat"
               → CLIP text encoder → 256-dim vector
```

Lightweight — runs instantly.

---

## 6. Part 3: Mask Decoder

```
image_embedding (64×64×256)
      +
prompt_embedding (256-dim)
      ↓
Transformer decoder (2 layers)
      ↓
3 candidate masks + confidence scores

Mask 1: whole object     confidence=0.92
Mask 2: part of object   confidence=0.78
Mask 3: larger region    confidence=0.61
      ↓
Pick highest confidence → final mask
```

Why 3 masks? Ambiguity — click on face → is mask the face, head, or whole person?

**Analogy:** You point at someone's face. SAM offers 3 options: just the face, the head, or the whole person. You pick.

---

## 7. Why SAM Is Zero-shot

SAM trained on **SA-1B dataset:**

```
11 million images
1.1 billion masks
```

Saw so many diverse objects → learned "what makes an object" — not specific classes.

**Analogy:** Surgeon who dissected 1 billion different things — can outline any new object cleanly without ever seeing it before.

---

## 8. Full Pipeline

```
Image
    ↓
Image Encoder (ViT-H) → image embedding  [run once]
    ↓
User clicks point (200, 150)
    ↓
Prompt Encoder → prompt embedding  [instant]
    ↓
Mask Decoder → 3 masks + scores  [instant]
    ↓
Best mask → draw on image
```

---

## 9. SAM vs YOLO Seg vs U-Net

| | U-Net | YOLO Seg | SAM |
|--|-------|----------|-----|
| Classes | Fixed | Fixed | Any |
| Prompt | None | None | Point/Box/Text |
| Speed | Medium | Real-time | Slow (image encoder) |
| Zero-shot | No | No | Yes |
| Training needed | Yes | Yes | No (use as-is) |
| Use case | Medical | Real-time | Interactive, zero-shot |

---

## 10. SAM 2 (2024)

Extended SAM to **videos.**

```
Click object in frame 1 → SAM 2 tracks + segments across all frames
```

Same architecture + memory module for temporal tracking.

---

## 11. Code

```python
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image (runs image encoder once)
image = cv2.imread("image.jpg")
predictor.set_image(image)

# Point prompt
masks, scores, _ = predictor.predict(
    point_coords=np.array([[200, 150]]),  # click point
    point_labels=np.array([1]),           # 1=foreground, 0=background
    multimask_output=True                 # return 3 candidate masks
)

# Best mask
best_mask = masks[np.argmax(scores)]
```

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Prompt-based | Point/box → mask | Point at object → outline it |
| Image encoder | ViT-H → 64×64×256 embedding | High-quality scene scan done once |
| Prompt encoder | Point/box → 256-dim vector | Convert click to numbers |
| Mask decoder | Both embeddings → 3 masks | 3 outline options, pick best |
| Zero-shot | 1.1B masks training | Surgeon who dissected 1B things |
| SAM 2 | Adds video tracking | Follows object across frames |

---

## One-line Summary

> SAM = give a point or box prompt → image encoder (ViT-H) understands scene → mask decoder combines image + prompt → outputs pixel-perfect mask for any object, zero-shot.
