# Day 16 — Transformer Architecture

## Concepts Covered
- Full encoder-decoder architecture
- Input embedding + positional encoding
- Self-attention (encoder)
- Add & Norm (residual + LayerNorm)
- Feed Forward Network
- Masked self-attention (decoder)
- Cross-attention (decoder)
- Encoder-only vs Decoder-only vs Encoder-Decoder

---

## 1. Big Picture

```
Original use: translation ("Hello" → "Bonjour")

Encoder: reads + understands input sentence
Decoder: generates output sentence word by word
```

---

## 2. Full Architecture

```
Input sentence
      ↓
[Input Embedding + Positional Encoding]
      ↓
┌─────────────────────┐
│   ENCODER (×6)      │
│  Self-Attention     │
│  Add & Norm         │
│  Feed Forward       │
│  Add & Norm         │
└─────────────────────┘
      ↓
   encoder output (context)
      ↓
┌─────────────────────┐
│   DECODER (×6)      │
│  Masked Self-Att    │
│  Add & Norm         │
│  Cross-Attention    │
│  Add & Norm         │
│  Feed Forward       │
│  Add & Norm         │
└─────────────────────┘
      ↓
  Linear + Softmax
      ↓
  Output word
```

---

## 3. Component 1: Input Embedding

Convert each word/token to a vector:

```
"cat" → [0.2, 0.8, 0.1, ...]  512-dim
"sat" → [0.9, 0.1, 0.3, ...]  512-dim
```

Learned during training — similar words → similar vectors.

---

## 4. Component 2: Positional Encoding

Transformer has no built-in order sense. Add position signal:

```
token_1_input = word_embedding_1 + position_encoding_1
token_2_input = word_embedding_2 + position_encoding_2
```

Fixed formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Analogy:** Words on numbered seats. Word knows its meaning AND its position.

---

## 5. Component 3: Self-Attention (Encoder)

```
Attention(Q,K,V) = softmax(QKᵀ/√d) × V
```

Every word attends to every other word → enriched context vectors.
(Full details in Day 15 notes)

---

## 6. Component 4: Add & Norm

After each sub-layer:

```
output = LayerNorm(x + sublayer(x))
```

**Add (residual):** Add original input back → prevents vanishing gradient (like ResNet).
**Norm (LayerNorm):** Normalize across features → stable training.

```
x → self-attention → attention_output
LayerNorm(x + attention_output) → next layer
```

**Analogy:** After group discussion (attention), each person combines new info WITH original opinion (residual). Then everyone recalibrates to same scale (norm).

---

## 7. Component 5: Feed Forward Network

Applied to each token independently:

```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2

Input:  512-dim
Hidden: 2048-dim  (4× expansion)
Output: 512-dim
```

**Why FFN after attention?**
- Attention = gather information from others (social step)
- FFN = process + transform that information (individual thinking step)

**Analogy:** After group discussion → each person privately processes their notes.

---

## 8. Component 6: Masked Self-Attention (Decoder)

Decoder generates output one word at a time.
Cannot look at future words:

```
Generating word 3:
  Can attend to:    word 1, word 2        ← allowed
  Cannot attend to: word 4, 5, 6...      ← masked (set to -∞)
```

Mask = upper triangle of attention matrix set to -∞ → after softmax → 0.

**Analogy:** Writing exam — can review what you already wrote, can't peek at future answers.

---

## 9. Component 7: Cross-Attention (Decoder)

Decoder attends to encoder output:

```
Q = from decoder  (what am I generating?)
K = from encoder  (what was the input?)
V = from encoder  (input information)
```

```
Generating "Bonjour":
  Q = "Bonjour" query
  K, V = encoder output of "Hello"
  → decoder attends to relevant input words
```

**Analogy:** Translator (decoder) glances at original speech (encoder output) while writing each translated word.

---

## 10. Full Data Flow (Translation Example)

```
Input: "The cat sat"

1. Embed + position encode → [the_vec+pos1, cat_vec+pos2, sat_vec+pos3]
2. Encoder × 6:
   self-attention → add&norm → FFN → add&norm
3. Encoder output: context vectors (full input understood)

4. Decoder input: <start> token
5. Masked self-attention (can't see future)
6. Cross-attention with encoder output
7. FFN → linear → softmax → "Le"

8. Feed "Le" back → generate "chat"
9. Feed "Le chat" back → generate "est"
...until <end> token
```

---

## 11. Encoder-Only vs Decoder-Only vs Encoder-Decoder

```
Encoder-Only:      BERT, DINOv2, ViT
  → understands input deeply
  → classification, embeddings, features
  → no generation

Decoder-Only:      GPT, LLaMA, Claude
  → generates text autoregressively
  → text generation, completion, chat
  → no separate encoder

Encoder-Decoder:   T5, BART, original Transformer
  → understands input → generates output
  → translation, summarization, Q&A
```

---

## Quick Reference

| Component | Role | Analogy |
|-----------|------|---------|
| Input embedding | Word → vector | Word to number |
| Positional encoding | Add position info | Numbered seats |
| Self-attention | Every token attends to every other | Everyone talks to everyone |
| Add & Norm | Residual + normalize | Keep original + recalibrate |
| Feed Forward | Per-token transformation | Private processing after group discussion |
| Masked self-attention | Can't see future tokens | Can't peek at future exam answers |
| Cross-attention | Decoder attends to encoder | Translator glances at original |

| Model Type | Examples | Good At |
|------------|---------|---------|
| Encoder-only | BERT, ViT, DINOv2 | Understanding, features |
| Decoder-only | GPT, LLaMA, Claude | Generating text |
| Encoder-decoder | T5, BART | Translation, summarization |

---

## One-line Summary

> Transformer = encoder (self-attention + FFN × 6) understands input → decoder (masked self-attention + cross-attention + FFN × 6) generates output word by word, attending to both previous outputs and encoder context.
