# Day 11 — Vision Transformer (ViT)

## Concepts Covered
- Why ViT over CNN
- Patch embedding
- Positional encoding
- CLS token
- Self-attention
- CNN vs ViT comparison
- Task-wise usage

---

## 1. Problem With CNNs

CNN uses local kernels — each pixel only sees its small neighbourhood.
To understand global context, CNN needs many layers.

**Analogy:** CNN reads a book word by word, slowly building context. Can't jump to page 100 to connect with page 1.

---

## 2. ViT Key Idea — Global Context in One Shot

Every patch talks to every other patch directly in one step.

**Analogy:** Everyone in a room talks to everyone else simultaneously — no waiting.

---

## 3. Full Pipeline

```
224×224 image
    ↓
Split into 196 patches (16×16 each)
    ↓
Patch embedding (linear layer → 768-dim vector per patch)
    ↓
Add positional encoding + CLS token → 197 × 768
    ↓
Transformer encoder (12 layers of self-attention)
    ↓
Take CLS token → 768 (context vector = summary of whole image)
    ↓
Linear → softmax → class
```

---

## 4. Step-by-Step

### Step 1: Split Into Patches

```
224×224 image → 14×14 grid of 16×16 patches = 196 patches

Each patch = 16×16×3 = 768 numbers → flattened to vector
```

**Analogy:** Cut photo into 196 tiles. Each tile = one "word" for the transformer.

---

### Step 2: Patch Embedding

```
Patch (768 numbers) → Linear layer → Embedding (768-dim vector)
```

Converts each patch into a vector the transformer can process — like word embeddings in NLP.

---

### Step 3: Positional Encoding

Transformer has no built-in sense of position. Add learnable position vector to each patch:

```
patch_1_embedding + position_1 → patch 1 knows it's top-left
patch_2_embedding + position_2 → patch 2 knows it's next
```

**Analogy:** Numbers written on tile backs — even if shuffled, transformer knows original position.

---

### Step 4: CLS Token

Special token added at start:

```
Input sequence: [CLS], P1, P2, P3, ..., P196  → 197 tokens
```

After attention, CLS token collects info from ALL patches → becomes context vector.

**Analogy:** Class president. After everyone discusses, president summarizes the group decision.

---

### Step 5: Self-Attention

```
Q = Query  (what am I looking for?)
K = Key    (what do I contain?)
V = Value  (what information do I give?)

Attention = softmax(Q × Kᵀ / √d) × V
```

Every patch attends to every other patch directly — no distance limit.

```
Example — Cat image:
Patch 1 (cat face) → directly attends to Patch 150 (sofa)
Attention score = 0.7  (high — they are related)
Patch 1 immediately knows: "I am on a sofa"
```

**Analogy:** Each tile asks "which other tiles are most relevant to me?" and borrows information from them directly.

---

### Step 6: Classification

```
Transformer output: 197 × 768
    ↓
Take CLS token: 768 numbers (context vector)
    ↓
Linear layer → num_classes → softmax → class
```

---

## 5. CNN vs ViT — Core Difference

```
CNN:
  Kernel = 3×3 → sees only neighbours
  Patch 1 ↔ Patch 196 connection → needs many layers
  Context builds slowly, layer by layer

ViT:
  Self-attention → patch 1 directly talks to patch 196
  Complete global context in ONE layer
  Every patch knows every other patch immediately
```

| | CNN | ViT |
|--|-----|-----|
| Input | Pixels | Patches |
| Context | Local (kernel size) | Global (all patches) |
| Position | Built-in (spatial) | Added (positional encoding) |
| Data needed | Less | More |
| Speed | Fast | Slower |
| Accuracy (large data) | Good | Better |
| Multimodal | Hard | Easy |

---

## 6. NLP Analogy

```
Patches          = words in a sentence
Patch embedding  = word embedding
Positional enc   = word position (1st, 2nd, 3rd...)
Self-attention   = each word looks at all others for context
CLS token        = [summary] token = final sentence meaning
```

**ViT = BERT for images.** Same architecture, image patches instead of words.

---

## 7. Task-wise Usage

### Classification — ViT's Home Ground

ViT was designed for classification. CLS token → class directly.

**When:** Large dataset, fine-grained classification, scene understanding.
**Models:** ViT-B/16, ViT-L/32, DeiT

---

### Detection — ViT as Backbone

ViT alone can't detect. Use as backbone + detection head.

```
Image → ViT → rich feature maps → DETR decoder → bbox + class
```

**DETR:** No anchor boxes. Directly predicts bbox + class.
**When:** Complex scenes, object relationships matter.
**Weakness:** Slower than YOLO for real-time.

---

### Segmentation — ViT + Decoder

```
Image → ViT → patch embeddings → decoder → pixel mask
```

**Models:**
- SETR — ViT + simple decoder
- Segmenter — ViT + mask transformer
- SAM — ViT image encoder + prompt encoder + mask decoder

**When:** High accuracy, zero-shot segmentation (SAM).

---

### Multimodal — ViT's Biggest Strength

```
Image patches → ViT → image embedding
Text tokens   → BERT → text embedding
                   ↓
             compare / combine
```

**Models:** CLIP, BLIP, LLaVA, Flamingo

ViT patch embeddings align naturally with text embeddings — this is where ViT beats CNN completely.

---

## 8. Simple Rule

```
Speed needed          → CNN / YOLO
Accuracy + large data → ViT
Image + text combined → ViT (must)
Zero-shot anything    → ViT-based (CLIP, SAM)
Small dataset         → CNN
Real-time edge device → CNN / YOLO
```

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Patches | 16×16 tiles → 768-dim vectors | Photo cut into tiles = words |
| Patch embedding | Linear layer | Translate tile to transformer language |
| Positional encoding | Learnable position vector added | Numbers on tile backs |
| CLS token | Summary token → context vector | Class president summarizes discussion |
| Self-attention | Every patch × every patch | Everyone talks to everyone simultaneously |
| Global context | Complete in one layer | Read entire book at once |

| Task | Use ViT? | Why |
|------|----------|-----|
| Classification | Best | Global context, CLS token |
| Detection | Yes (backbone) | DETR, slower than YOLO |
| Segmentation | Yes (with decoder) | SAM uses ViT |
| Multimodal | Essential | Patch embeddings align with text |
| Real-time edge | No | Too slow |
| Small dataset | No | Needs large data |
