# Day 24 — Multimodal LLMs

## Concepts Covered
- What is Multimodal LLM
- How image gets into LLM
- LLaVA architecture (Vision Encoder + Projector + LLM)
- What is the Projector
- Popular multimodal models
- Code — LLaVA with Transformers
- Code — Claude Vision API
- CV Engineer use cases
- YOLO + LLaVA pipeline (Portfolio Project 3)

---

## 1. What is Multimodal LLM?

Normal LLM: only understands text.

**Multimodal LLM = understands both image + text together.**

```
Input:  [image of a cat] + "What is in this image?"
Output: "A cat sitting on a mat"

Input:  [product defect image] + "Is this defect critical?"
Output: "Yes, surface crack visible on top-left corner. Severity: HIGH"
```

**Analogy:** Normal LLM = person who can only read text. Multimodal LLM = person who can look at a photo AND read text, then answer about both together.

---

## 2. How Image Gets Into LLM

LLM only understands tokens (numbers). Image must be converted to tokens first.

```
Image (pixels)
    ↓
Vision Encoder (ViT)     ← converts image to feature vectors
    ↓
Projector / Adapter      ← converts vision features → language token space
    ↓
LLM                      ← sees image tokens + text tokens together
    ↓
Text output
```

---

## 3. Architecture — LLaVA

**LLaVA = Large Language and Vision Assistant**

```
┌─────────────────────────────────────────────┐
│                   LLaVA                     │
│                                             │
│  [Image] → CLIP ViT → Projector ──┐        │
│                                    ↓        │
│  [Text]  → Tokenizer ──────────→ LLM → Answer│
│                                             │
└─────────────────────────────────────────────┘
```

3 components:
```
1. Vision Encoder  → CLIP ViT (converts image → 256 feature vectors)
2. Projector       → Linear layer (maps vision space → LLM token space)
3. LLM             → Vicuna / LLaMA (generates text answer)
```

---

## 4. Projector — The Bridge

Vision encoder output and LLM input are in different spaces:

```
CLIP ViT output:  256 × 1024 dim  (vision space)
LLM input:        N × 4096 dim    (language space)
```

Projector = simple linear layer that maps one space to the other:

```
256 × 1024  →  Projector  →  256 × 4096

Now image = 256 "visual tokens" in same space as text tokens
LLM sees them the same way as word tokens
```

**Analogy:** CLIP speaks French (vision language), LLM speaks English (text language). Projector = translator between them.

---

## 5. Popular Multimodal LLMs

| Model | Vision Encoder | LLM | Best For |
|-------|---------------|-----|----------|
| LLaVA 1.6 | CLIP ViT-L | Vicuna/LLaMA | Open source, local |
| Qwen-VL | ViT | Qwen | Multilingual, documents |
| GPT-4V | Unknown | GPT-4 | Best quality, API |
| Claude 3 | Unknown | Claude | Best reasoning |
| Gemini | Unknown | Gemini | Google ecosystem |
| InternVL | InternViT | InternLM | Strong open source |

---

## 6. Code — LLaVA with Transformers

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

# ── 1. Load Model ─────────────────────────────────────────
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# ── 2. Load Image ─────────────────────────────────────────
image = Image.open("product_defect.jpg")

# ── 3. Prepare Input ──────────────────────────────────────
prompt = "[INST] <image>\nIs there a defect in this product? Describe it. [/INST]"
inputs = processor(prompt, image, return_tensors="pt").to("cuda")

# ── 4. Generate ───────────────────────────────────────────
output = model.generate(**inputs, max_new_tokens=200)
answer = processor.decode(output[0], skip_special_tokens=True)

print(answer)
# "Yes, there is a visible crack on the top-left surface of the product.
#  The crack measures approximately 2-3mm. Severity: HIGH. Recommended action: REJECT"
```

---

## 7. Code — Claude Vision API (Easier)

```python
import anthropic
import base64

client = anthropic.Anthropic(api_key="your-key")

# Load and encode image
with open("product_defect.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Is there a defect in this product? Describe severity."
                }
            ],
        }
    ],
)

print(response.content[0].text)
```

---

## 8. CV Engineer Use Cases

Since you have CV background, multimodal LLMs are powerful for:

```
1. Defect detection + explanation
   YOLO detects defect location → LLaVA explains what the defect is

2. Quality report generation
   Image in → LLM writes structured quality report out

3. Visual Q&A
   Inspector asks "is this weld clean?" → model answers yes/no + reason

4. Document understanding
   Invoice/form image → extract structured data as JSON

5. Medical imaging
   X-ray + "any abnormalities?" → LLM describes findings
```

---

## 9. YOLO + LLaVA Pipeline (Portfolio Project 3)

```
Image input
    ↓
YOLOv8 → detects objects + bounding boxes
    ↓
Crop each detected region
    ↓
LLaVA → describe each cropped region in detail
    ↓
Claude → write structured report
    ↓
Output: JSON report with detections + descriptions
```

```python
from ultralytics import YOLO
from PIL import Image

yolo = YOLO("yolov8n.pt")
results = yolo("factory_floor.jpg")

for box in results[0].boxes:
    # Crop detected region
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cropped = Image.open("factory_floor.jpg").crop((x1, y1, x2, y2))
    cropped.save("region.jpg")

    # LLaVA describes it
    description = llava_describe("region.jpg", "Describe this object and any defects")
    print(f"Object at ({x1},{y1}): {description}")
```

---

## Summary

```
Multimodal LLM  = understands image + text together
Architecture    = Vision Encoder → Projector → LLM
Vision Encoder  = CLIP ViT (image → feature vectors)
Projector       = linear layer (vision space → language space)
LLM             = generates text answer

Popular models:
  LLaVA    → open source, local
  Qwen-VL  → multilingual
  GPT-4V   → best API quality
  Claude 3 → best reasoning

Your use case:
  YOLO (detect) + LLaVA (describe) + Claude (report) = powerful CV pipeline
```
