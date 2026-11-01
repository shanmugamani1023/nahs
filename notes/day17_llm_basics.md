# Day 17 — LLM Basics

## Concepts Covered
- What is an LLM
- Tokenization + BPE
- Next token prediction
- Autoregressive generation
- Logits → probabilities
- Temperature
- Sampling strategies (Greedy, Top-K, Top-P)
- Context window
- Model parameters

---

## 1. What is an LLM?

A large decoder-only transformer trained to predict the next token.

```
Input:  "The cat sat on the"
Output: "mat"  ← predicted next token
```

Everything else (reasoning, coding, chat) emerges from doing this extremely well on massive data.

**Analogy:** LLM = someone who read the entire internet. When you give them a sentence, they complete it based on everything they've read.

---

## 2. Tokenization — Text to Numbers

Model doesn't see words — sees **tokens** (numbers).

```
"Hello world" → [15496, 995]

"unhappy" → ["un", "happy"]    ← 2 tokens
"cat"     → ["cat"]            ← 1 token
"🐱"      → 4 tokens
```

Uses **BPE (Byte Pair Encoding)** — splits text into subword units.

**Why subwords?**
```
Whole words:  vocabulary = millions (too large)
Characters:   sequences = very long (too slow)
Subwords:     balance — 30,000–100,000 token vocabulary
```

**Analogy:** Instead of memorizing every English word — learn common syllables. "un" + "happy" = "unhappy". Small building blocks, infinite combinations.

**Token count rule of thumb:**
```
1 token ≈ 4 characters ≈ 0.75 words
100 words ≈ 133 tokens
1 page    ≈ 500 tokens
```

---

## 3. Next Token Prediction — Training

```
Training text: "The cat sat on the mat"

Input: "The cat sat on the" → predict → "mat"  ✓
Input: "The cat sat on"     → predict → "the"  ✓
Input: "The cat sat"        → predict → "on"   ✓
```

Each text = many training examples automatically. No labels needed.

Loss = cross-entropy between predicted probabilities and actual next token.

**Analogy:** Child learns language by hearing complete sentences. LLM does this for billions of sentences.

---

## 4. Inference — Autoregressive Generation

```
Prompt: "The cat"

Step 1: "The cat" → model → probabilities → sample "sat"
Step 2: "The cat sat" → model → probabilities → sample "on"
Step 3: "The cat sat on" → model → probabilities → sample "the"
...
Repeat until <end> token or max length
```

Each step = one full forward pass through transformer.

---

## 5. Logits → Probabilities

Raw model output = **logits** (unnormalized scores):

```
Logits: [2.1, 0.5, -1.2, 3.4, 0.8, ...]  ← one per vocab token
         ↓ softmax
Probs:  [0.15, 0.04, 0.01, 0.42, 0.06, ...]  ← sum = 1
```

---

## 6. Temperature — Control Randomness

```
adjusted_logits = logits / T
```

```
T = 0.1  (low — deterministic):
  probs = [0.98, 0.01, 0.00, 0.00...]  ← winner takes all

T = 1.0  (default):
  probs = [0.42, 0.15, 0.06, 0.04...]  ← normal

T = 2.0  (high — random):
  probs = [0.25, 0.22, 0.20, 0.18...]  ← flat, creative
```

**Analogy:**
- Low temp = confident expert, always picks best answer
- High temp = creative, picks surprising answers
- T=0 = always greedy, always same output

---

## 7. Sampling — How to Pick the Next Token

Model outputs probabilities for every possible next token:

```
"mat"    → 0.42
"floor"  → 0.25
"sofa"   → 0.15
"roof"   → 0.10
"moon"   → 0.05
"xyz"    → 0.001
```

**Sampling = which token do we pick?**

---

### Greedy — Always Pick Highest

```
Always pick "mat" (0.42)
→ deterministic, same output every time
→ boring, repetitive
```

**Analogy:** Always order the same restaurant dish because it's most popular. Safe but boring.

---

### Random Sampling — Pick by Probability

```
42% chance → "mat"
25% chance → "floor"
15% chance → "sofa"
 5% chance → "moon"
 0.1% chance → "xyz"  ← garbage token can still be picked
```

**Problem:** Low probability garbage tokens can still be selected.

**Analogy:** Spin a weighted roulette wheel. Any section can win.

---

### Top-K — Limit the Pool

```
K = 3 → only consider top 3 tokens:

"mat"    → 0.42
"floor"  → 0.25
"sofa"   → 0.15
─────────────────
Renormalize → [0.51, 0.31, 0.18]
Sample randomly from these 3 only.
```

Cuts off bad tokens completely.

**Analogy:** Only pick from top 3 chef recommendations. Still random, but quality guaranteed.

---

### Top-P (Nucleus Sampling) — Smarter Limit

```
P = 0.9 → keep adding tokens until cumulative prob ≥ 0.9:

"mat"    0.42  cumsum = 0.42
"floor"  0.25  cumsum = 0.67
"sofa"   0.15  cumsum = 0.82
"roof"   0.10  cumsum = 0.92  ← stop (crossed 0.9)

Sample from: mat, floor, sofa, roof only.
```

**Why better than Top-K?**
```
Model very confident:
  "mat" = 0.95 → Top-P picks only "mat" (smart)
  Top-K=50 still considers 50 tokens (unnecessary)

Model uncertain:
  all tokens ≈ 0.02 → Top-P picks many tokens (creative)
  Top-K=50 still picks exactly 50
```

**Analogy:** Top-K = always shortlist exactly 3 candidates. Top-P = shortlist however many cover 90% of quality — sometimes 1, sometimes 10.

---

### Temperature + Sampling Together

```
Temperature = HOW SPREAD the probabilities are
Sampling    = HOW YOU PICK from those probabilities

Low temp + Top-P:  focused, picks from small likely pool
High temp + Top-P: creative, picks from larger pool
```

**Real settings:**
```
Chatbot (factual):   T=0.2, Top-P=0.9   → precise
Creative writing:    T=0.9, Top-P=0.95  → varied
Code generation:     T=0.1, Top-K=10    → deterministic
```

---

## 8. Context Window

Maximum tokens model can process at once:

```
GPT-3:   4,096 tokens
GPT-4:   128,000 tokens
Claude:  200,000 tokens
Gemini:  1,000,000 tokens
```

Beyond context window → model forgets earlier content.

**Analogy:** Short-term memory limit. Model can only think about this many tokens at once.

---

## 9. Parameters

```
GPT-2:    117M parameters
GPT-3:    175B parameters
LLaMA 3:  8B / 70B parameters
```

More parameters = more knowledge = better but slower + more memory.

```
7B model  → 8GB GPU  (consumer, RTX 3070)
70B model → 40GB+ GPU (professional, A100)
```

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Token | Subword unit → ID number | Building block syllable |
| Next token prediction | P(next \| context) | Complete the sentence |
| Autoregressive | Generate one token, feed back, repeat | Write word by word |
| Temperature | Divide logits by T | Confidence dial |
| Greedy | Always pick highest prob | Always order most popular dish |
| Top-K | Sample from K most likely | Pick from top 3 chef picks |
| Top-P | Sample until cumprob ≥ P | Cover 90% quality pool |
| Context window | Max tokens in one pass | Short-term memory limit |

---

## One-line Summary

> LLM = decoder-only transformer that predicts next token from context. Text → tokens → logits → temperature scaling → Top-K/Top-P sampling → next token → repeat until done.
