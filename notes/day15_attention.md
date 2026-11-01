# Day 15 — Attention Mechanism

## Concepts Covered
- Problem with RNN
- Q, K, V vectors
- Attention score computation
- Scaling
- Softmax
- Weighted sum → context vector
- Multi-head attention
- Self-attention vs cross-attention

---

## 1. Problem With RNN

```
"The cat sat on the mat because it was tired"

To understand "it" = "cat" → RNN must remember from 8 words ago.
Long sequences → memory fades → wrong connections.
```

**Attention:** Every word directly connects to every other word in one step.

---

## 2. Core Idea

```
"it" → looks at all words → "cat" gets highest score →
"it" now knows it refers to "cat"
```

**Analogy:** In a meeting, instead of passing notes one by one — everyone broadcasts to everyone simultaneously. Each person decides who to listen to most.

---

## 3. Q, K, V — The Three Vectors

For each input token, compute 3 vectors:

```
Q = Query   "What am I looking for?"
K = Key     "What do I contain?"
V = Value   "What information do I give if selected?"
```

**Analogy:** Library system.
- Q = your search query ("books about cats")
- K = book title/tags ("cat behavior", "dog training")
- V = actual book content
- Match query with keys → find relevant books → read their content

---

## 4. Create Q, K, V

Each input token × learned weight matrices:

```
token_embedding (512-dim)
    ↓
Q = embedding × W_Q  → 64-dim
K = embedding × W_K  → 64-dim
V = embedding × W_V  → 64-dim
```

W_Q, W_K, W_V = learned during training.

---

## 5. Full Attention Formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

### Step 1: Score
```
score(i, j) = Q_i · K_j   (dot product)

"it"  · "cat" = 8.5  (high — related)
"it"  · "mat" = 1.1  (low — not related)
"it"  · "the" = 0.3  (very low)
```

### Step 2: Scale
```
scaled_score = score / √d = score / √64 = score / 8
```
Prevents softmax saturation when dimensions are large.

### Step 3: Softmax
```
scores  = [8.5, 7.2, 1.1, 0.3]
weights = [0.52, 0.38, 0.06, 0.04]  ← sum = 1
```

### Step 4: Weighted Sum
```
output = 0.52×V_cat + 0.38×V_sat + 0.06×V_mat + 0.04×V_the
```

Output = enriched vector for "it" — now contains cat+sat context.

---

## 6. What Happens Per Word

```
Before attention:  "it" = just a pronoun (no context)
After attention:   "it" = knows it refers to "cat" (has context)
```

All words processed simultaneously:
```
"cat"   → attends to all → new enriched "cat" vector
"sat"   → attends to all → new enriched "sat" vector
"it"    → attends to all → new enriched "it" vector
"tired" → attends to all → new enriched "tired" vector
```

Each word's output = **context-aware representation** = attention vector.

---

## 7. Multiple Layers

```
Layer 1: basic relationships (cat → it)
Layer 2: deeper relationships (cat → tired, via it)
Layer 6: full sentence meaning understood
```

Each layer builds richer understanding on previous.

**Analogy:** Round 1 of meeting = basic info shared. Round 6 = full picture understood by everyone.

---

## 8. Multi-Head Attention

Run attention multiple times in parallel with different weight matrices:

```
Head 1: focuses on syntactic relationships (subject-verb)
Head 2: focuses on semantic relationships (pronoun-reference)
Head 3: focuses on positional relationships
...
Head 8: focuses on something else

All outputs concatenated → linear layer → final output
```

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O
```

**Analogy:** 8 experts in a meeting, each listening for different things. Combine all notes → richer understanding.

---

## 9. Self-Attention vs Cross-Attention

```
Self-attention:   Q, K, V from SAME sequence
                  "words attend to other words in same sentence"
                  Used in: encoder

Cross-attention:  Q from one sequence, K and V from another
                  "decoder attends to encoder output"
                  Used in: decoder (translation, captioning)
```

---

## 10. Attention in Vision (ViT)

Same formula, patches instead of words:

```
Patch_1 query · Patch_50 key = high score
→ cat face patch attends to cat tail patch directly
→ global context in one layer
```

---

## Quick Reference

| Step | Operation | Why |
|------|-----------|-----|
| Q, K, V | token × weight matrix | Learn what to look for / contain / share |
| Score | Q · Kᵀ | Measure relationship strength |
| Scale | ÷ √d | Prevent softmax saturation |
| Softmax | exp(x)/sum | Convert to probabilities (sum=1) |
| Weighted sum | weights × V | Mix information by relevance |

| Symbol | Meaning | Analogy |
|--------|---------|---------|
| Q | Query | Search query |
| K | Key | Book title/tags |
| V | Value | Book content |
| Attention weight | How relevant | How useful this book is |
| Output | Enriched vector | Summary from most relevant books |

---

## One-line Summary

> Every word computes Q, K, V → scores against all other words → softmax → weighted sum of values → enriched context vector that carries relationship information with all other words.
