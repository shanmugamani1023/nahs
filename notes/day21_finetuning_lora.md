# Day 21 — Fine-tuning + LoRA + QLoRA

## Concepts Covered
- What is fine-tuning
- Fine-tuning vs RAG
- Full fine-tuning vs LoRA
- How LoRA works (parallel layers, low rank)
- Why it's called Low Rank Adaptation
- Embedding dimension (8192 features per token)
- QLoRA = Quantization + LoRA
- Data format
- Code example

---

## 1. What is Fine-tuning?

Base LLM knows everything general. But it doesn't know your specific domain, tone, or task format.

**Fine-tuning = train the model further on YOUR data so it behaves exactly how you want.**

**Analogy:** A medical student graduates knowing general medicine. Then they do a specialization (residency) in cardiology. They don't re-learn everything — just specialize on top of existing knowledge. Fine-tuning is that residency.

---

## 2. Fine-tuning vs RAG

| Situation | Use |
|-----------|-----|
| Need domain-specific knowledge (private docs) | RAG |
| Data changes frequently | RAG |
| Need specific output format/style | Fine-tune |
| Need specific tone (customer support bot) | Fine-tune |
| Small context window, lots of examples | Fine-tune |
| Want faster inference (no retrieval step) | Fine-tune |
| Both knowledge + style needed | RAG + Fine-tune |

```
"What should it know?"  → RAG
"How should it behave?" → Fine-tune
```

---

## 3. Full Fine-tuning vs LoRA

### Full Fine-tuning
Update ALL parameters in the model.

```
LLaMA 3 8B → 8 billion parameters
All 8B updated every training step
GPU memory needed: ~80GB  ← needs A100
Cost: Very expensive
```

### LoRA — Low Rank Adaptation

**Don't update all 8B weights. Add small trainable parallel layers alongside specific layers.**

**Analogy:** You have a huge whiteboard (the LLM). Instead of erasing and rewriting the whole thing, you stick a small sticky note alongside it with your changes. The original whiteboard stays frozen. Only the sticky note gets updated.

---

## 4. How LoRA Works

### Parallel layers (NOT on top, alongside):

```
         ┌──── W (frozen, original) ────┐
input ───┤                               ├──→ output (sum)
         └──── A → B (tiny, trains) ────┘

Final output = W(input) + BA(input)
             = original + LoRA adjustment
```

NOT stacked:
```
input → W → A → B → output   ← WRONG
```

Parallel and added:
```
input → W ──────────────→ +  → output
input → A → B ──────────→ +
```

### Inside one Transformer Block:

```
┌─────────────────────────────────────┐
│  Attention:                         │
│                                     │
│  input → q_proj(frozen) ──────────→ + → Q
│  input → A→B (LoRA, trains) ──────→ +         
│                                     │
│  input → k_proj(frozen) ──────────→ K  (no LoRA)
│                                     │
│  input → v_proj(frozen) ──────────→ + → V
│  input → A→B (LoRA, trains) ──────→ +
│                                     │
│  Q,K,V → attention → output        │
└─────────────────────────────────────┘
```

### What are q_proj, v_proj?

These are the weight matrices that create Q and V in attention (same Q, K, V from Day 15):

```
input → q_proj (weight matrix) → Q
input → k_proj (weight matrix) → K
input → v_proj (weight matrix) → V

_proj = projection = linear layer = weight matrix
```

LoRA targets q_proj and v_proj because:
```
q_proj → controls WHAT the model searches for
v_proj → controls WHAT information gets passed forward
These 2 most influence model behavior/output style
```

### LoRA repeats across ALL blocks:

```
Transformer block 1:  q_proj ✅  v_proj ✅
Transformer block 2:  q_proj ✅  v_proj ✅
...
Transformer block 32: q_proj ✅  v_proj ✅

32 blocks × 2 layers = 64 LoRA pairs total
```

### Parameter count:

```
A = small matrix (4096 × 8)
B = small matrix (8 × 4096)

Params per pair = 2 × 4096 × 8 = 65K
64 pairs × 65K = ~4M trainable params

vs original: 8,000,000,000 params
Trainable = 0.05% only
```

---

## 5. Why "Low Rank"?

**Rank = how much information a matrix carries.**

```
q_proj weight matrix W:
  Shape = 4096 × 4096
  Rank  = 4096  (full rank)

LoRA update BA:
  A shape = 4096 × 8
  B shape = 8    × 4096
  BA shape = 4096 × 4096  (same as W)
  BUT rank of BA = 8 only  ← LOW RANK
```

**Why low rank is enough:**

Research found that the actual change needed during fine-tuning is naturally low rank. The model doesn't need 4096 dimensions of change — just 8 dimensions is enough to adapt behavior.

```
Full rank update = overkill for fine-tuning
Low rank update  = just enough, way cheaper
```

**Analogy:** To change a car's color, you don't rebuild the entire engine. The "change" is simple (low complexity = low rank). Repainting is enough.

### Full name breakdown:
```
Low        → small rank (r=8, not 4096)
Rank       → mathematical rank of update matrix BA
Adaptation → specializing model to new task
```

---

## 6. Embedding Dimension (8192 features)

Each token is represented as a vector of N numbers (embedding dimension):

```
"cat" → [0.2, -0.5, 0.8, 0.1, ... ]  ← N numbers
"dog" → [0.3, -0.4, 0.9, 0.2, ... ]  ← N numbers
```

Different models:
```
Model              Embedding dim
─────────────────────────────────
LLaMA 3 8B    →   4096
LLaMA 3 70B   →   8192
GPT-4         →   ~12288 (estimated)
BERT base     →   768
```

q_proj shape = embedding_dim × embedding_dim

---

## 7. QLoRA = Quantization + LoRA

### Problem with LoRA alone:
```
LLaMA 3 8B weights in 16-bit:
8B params × 2 bytes = 16GB just to load
+ training overhead = ~24GB total
→ Needs RTX 3090/4090
```

### Quantization = store weights in fewer bits:
```
16-bit → 4-bit  (4x less memory)

16-bit: 0.234567  (precise)
4-bit:  0.25      (approximate, close enough)

8B × 0.5 bytes = 4GB to load model
```

### What gets quantized:
```
Base model weights (W) → 4-bit  ← quantized, frozen
LoRA weights (A, B)    → 16-bit ← stays full precision, trains
```

A, B stay 16-bit because training needs precise gradients.

### QLoRA flow:
```
Step 1: Load model weights in 4-bit  (W = 4-bit, frozen)
Step 2: Add LoRA layers A, B         (16-bit, will train)
Step 3: Train only A, B              (W never changes)
Step 4: Save only A, B weights       (~10MB file)
```

### Memory comparison:
```
Full fine-tune  →  80GB   (A100, expensive)
LoRA            →  24GB   (RTX 3090/4090)
QLoRA           →  12GB   (Free Colab T4) ✅
```

**Analogy:** Same renovation plan (LoRA). But first you compress the original blueprint from A3 to A5 size (quantization) to save desk space. You still make the same changes — just working with a smaller copy of the original.

---

## 8. Data Format for Fine-tuning

```json
[
  {
    "instruction": "Classify this defect image",
    "input": "Surface has a crack of 2mm width",
    "output": "Defect type: CRACK, Severity: HIGH"
  },
  {
    "instruction": "Describe the quality issue",
    "input": "Object shows discoloration on top surface",
    "output": "Quality issue: DISCOLORATION, Location: TOP, Action: REJECT"
  }
]
```

Minimum data needed:
| Task | Min Examples |
|------|-------------|
| Style/tone change | 50-100 |
| Classification | 200-500 |
| Domain Q&A | 500-1000 |
| Complex reasoning | 1000+ |

---

## 9. Code — LoRA with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 2. LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # rank of A and B matrices
    lora_alpha=32,    # scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # which layers to apply LoRA
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 1,572,864 || all params: 1,236,334,592 || trainable%: 0.13%

# 4. Train
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
)

trainer.train()

# 5. Save only LoRA weights (small file ~10MB vs 8GB)
model.save_pretrained("./my-lora-weights")
```

---

## Summary

```
Fine-tuning   = specialize model on your data
Full FT       = update all params → expensive (80GB GPU)
LoRA          = parallel A,B matrices on q_proj + v_proj → 24GB GPU
QLoRA         = 4-bit base model + LoRA → 12GB GPU (free Colab)

LoRA key idea:
  - NOT stacked on top, runs PARALLEL to original layer
  - Output = W(input) + BA(input)
  - Only A, B train — W stays frozen
  - 64 LoRA pairs across 32 transformer blocks
  - Trainable params = 0.1% of model

When to use:
  RAG   → "what to know"
  LoRA  → "how to behave / format / style"
```
