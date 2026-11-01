# Day 12 — CLIP (Contrastive Language-Image Pre-training)

## Concepts Covered
- Problem with normal classifiers
- Contrastive learning
- Image + Text encoders
- Similarity matrix
- Zero-shot classification
- Fine-tuning vs zero-shot
- Images needed per class
- GPU requirements

---

## 1. Problem With Normal Classifiers

```
Normal classifier: trained on fixed classes (cat, dog, car)
New class → retrain from scratch

CLIP: give image → compare with ANY text → no retraining needed
```

---

## 2. Core Idea — Train With Image + Label Together

```
Training:
  Give image + label together
      ↓
  Image encoder learns image features
  Text encoder learns label features
      ↓
  Loss forces same concept → same direction in 512-dim space
      ↓
  Similarity map built (which image ↔ which text)

Inference:
  New image → image encoder → embedding
      ↓
  Compare with all trained text embeddings
      ↓
  Closest text embedding → that's the label
```

---

## 3. Architecture

```
Image → ViT (Image Encoder) → 512-dim image embedding
                                        ↓
                              cosine similarity matrix
                                        ↑
Text  → Transformer (Text Encoder) → 512-dim text embedding
```

Both encoders trained jointly with contrastive loss.

---

## 4. Training — Contrastive Loss

Batch of N image-text pairs:

```
Images:  [I1, I2, I3, ... IN]
Texts:   [T1, T2, T3, ... TN]

Correct pairs: (I1,T1), (I2,T2)... → diagonal → should be HIGH
Wrong pairs:   (I1,T2), (I1,T3)... → off-diagonal → should be LOW
```

Similarity matrix:

```
                "bottle"  "pen"   "laptop"
image1(bottle)  [ 0.85    0.10    0.05  ]  ← diagonal HIGH ✓
image2(pen)     [ 0.08    0.88    0.04  ]
image3(laptop)  [ 0.06    0.05    0.89  ]
```

Loss = maximize diagonal, minimize off-diagonal → backprop → update BOTH encoders.

**Analogy:** 4 photos, 4 captions shuffled. Penalty for wrong matches. Train until photos always find correct caption.

---

## 5. What Each Encoder Learns

```
Image Encoder → learns "what bottle LOOKS like" visually
Text Encoder  → learns "what bottle MEANS in text"
Loss          → forces both → same direction in 512-dim space
```

Before training:
```
"a photo of a bottle" → random direction
"a photo of a pen"    → random direction
```

After training:
```
"a photo of a bottle" → pulled toward bottle images
"a photo of a pen"    → pulled toward pen images
```

---

## 6. Embedding Space

```
Training result:

  [bottle image 1] [bottle image 2] [new bottle image]
  [text "bottle"]                        ← all clustered together

  [pen image 1] [pen image 2]
  [text "pen"]                           ← clustered, far from bottles
```

Same concept (image + text) → same direction in space.

**Analogy:** After training, same-topic books on same shelf. Image book and text book of "cat" → same shelf.

---

## 7. Inference — Zero-shot Classification

```
Step 1: New image → Image Encoder → image_embedding

Step 2: All class texts → Text Encoder → text_embeddings
  "a photo of a bottle" → text_embedding_1
  "a photo of a pen"    → text_embedding_2
  "a photo of a laptop" → text_embedding_3

Step 3: Cosine similarity
  sim(image, text_1) = 0.92  ← highest
  sim(image, text_2) = 0.05
  sim(image, text_3) = 0.02

Step 4: Pick highest → "bottle"
```

No retraining. Just write new text labels.

---

## 8. Cosine Similarity

```
similarity = (A · B) / (|A| × |B|)

= 1.0  → same direction → perfect match
= 0.0  → perpendicular → no relation
= -1.0 → opposite → completely different
```

**Analogy:** Two arrows pointing same direction = similar. Perpendicular = unrelated.

---

## 9. New Class — What Happens

### Pre-trained CLIP (400M pairs)
```
New class likely already in training data
→ zero-shot works, no retraining needed
```

### Custom CLIP (10 classes only)
```
New class never seen
→ encoder doesn't know its features
→ must retrain
```

**Analogy:** OpenAI CLIP = student who read entire internet. Your custom CLIP = student who studied 10 topics only.

---

## 10. Images Needed Per Class

| Approach | Images Per Class | When |
|----------|-----------------|------|
| Zero-shot (pre-trained CLIP) | 0 | Common objects, internet concepts |
| Few-shot | 1–10 | Quick prototype, limited data |
| Fine-tune pre-trained CLIP | 50–1000 | Custom domain, similar to internet |
| Very different domain (medical) | 1000–5000 | X-ray, satellite, microscope |
| Train from scratch | 10,000+ | Never recommended |

**Rule:** Always start with pre-trained CLIP + few-shot. Only fine-tune if accuracy not enough.

---

## 11. Few-shot Technique

```
5 example images per class → encode → average = class prototype

New image → encode → compare with prototypes → classify
```

No retraining of encoder. Just build prototypes from examples.

---

## 12. GPU Requirements

| Task | GPU Needed |
|------|-----------|
| Inference (single image) | CPU fine |
| Inference (real-time/batch) | GPU needed |
| Fine-tuning ViT-B/32 | 8GB VRAM minimum |
| Fine-tuning ViT-L/14 | 16GB+ VRAM |

```python
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
```

**Free GPU options:** Google Colab (T4), Kaggle (P100)

---

## 13. Model Sizes

```
Model        Parameters   VRAM    Speed
ViT-B/32     151M         4GB     fastest
ViT-B/16     151M         5GB     fast
ViT-L/14     428M         10GB    slow
ViT-L/14@336 428M         16GB    slowest, most accurate
```

---

## 14. Use Cases

| Use Case | How |
|----------|-----|
| Zero-shot classification | Compare image with class name texts |
| Image search | Find images matching text query |
| Text-to-image (DALL-E, SD) | CLIP guides image generation |
| Anomaly detection | "normal product" vs "defective product" |
| Multimodal RAG | Image + text retrieval together |

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Contrastive learning | Maximize correct pairs, minimize wrong | Reward correct caption match |
| Image encoder | ViT → 512-dim vector | Photo → summary number |
| Text encoder | Transformer → 512-dim vector | Caption → summary number |
| Cosine similarity | Dot product of unit vectors | Arrows pointing same direction |
| Zero-shot | No retraining, just new text labels | New question, same knowledge |
| Embedding space | Same concept clusters together | Same topic books on same shelf |

---

## One-line Summary

> Train with image+label pairs → build shared 512-dim embedding space where same concept (image+text) points same direction → new image finds closest text embedding → that's the prediction.
