# Day 18 — LLM APIs

## Concepts Covered
- What is an LLM API
- Claude API + OpenAI API
- Message roles (system, user, assistant)
- System prompt
- Prompt engineering techniques
- Structured output (JSON)
- Key parameters
- Token cost
- Streaming
- Vision (sending images)
- Base64 encoding

---

## 1. What is an LLM API?

Instead of running model locally (needs 40GB GPU) — call a hosted model via HTTP.

```
Your code → API request → Anthropic/OpenAI servers → response
```

Pay per token used. No GPU needed.

---

## 2. Claude API — Basic Call

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is a transformer?"}
    ]
)

print(message.content[0].text)
```

---

## 3. OpenAI API — Basic Call

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What is a transformer?"}
    ]
)

print(response.choices[0].message.content)
```

Both APIs follow same pattern — almost identical.

---

## 4. Message Roles

```
system    → instructions for the model (behavior, persona, rules)
user      → human input
assistant → model response
```

```python
messages = [
    {"role": "system",    "content": "You are a CV engineer expert."},
    {"role": "user",      "content": "Explain YOLO"},
    {"role": "assistant", "content": "YOLO is..."},   # previous response
    {"role": "user",      "content": "How does NMS work?"}  # follow-up
]
```

Full conversation history passed every time → model has context.

---

## 5. System Prompt — Most Important

Controls model behavior completely:

```python
system = """
You are an expert CV engineer assistant.
- Always give technical explanations with analogies
- Use code examples when relevant
- Be concise
- If you don't know something, say so
"""
```

**Analogy:** Job description for the model. Before answering anything, it reads this and acts accordingly.

---

## 6. Prompt Engineering — Key Techniques

### Be Specific
```
Bad:  "Explain YOLO"
Good: "Explain YOLOv8 anchor-free detection head in 3 bullet points
       with a code example"
```

### Give Context
```
"I am a CV engineer with 4 years experience.
 Explain LoRA fine-tuning assuming I know PyTorch."
```

### Few-shot Examples
```
"Classify these as positive/negative:

Example 1: 'Great product' → positive
Example 2: 'Terrible quality' → negative

Now classify: 'Works as expected'"
```

### Chain of Thought
```
"Solve this step by step:
 An image is 416×416. YOLO uses 32× downsampling.
 What is the feature map size?"
```

### Output Format
```
"Respond in JSON format:
{
  'answer': '...',
  'confidence': 0-1,
  'reasoning': '...'
}"
```

---

## 7. Structured Output — Force JSON

```python
message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": """Extract info from this image description.
        Return JSON: {"object": "...", "color": "...", "count": 0}

        Description: Three red apples on a table."""
    }]
)

import json
result = json.loads(message.content[0].text)
print(result["object"])  # "apple"
print(result["count"])   # 3
```

---

## 8. Key Parameters

```python
client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,    # max output tokens
    temperature=0.7,    # randomness (0-1)
    top_p=0.9,          # nucleus sampling
    system="...",       # system prompt
    messages=[...]      # conversation history
)
```

---

## 9. Token Cost Awareness

```
Claude Sonnet:
  Input:  $3 per million tokens
  Output: $15 per million tokens

1 API call (1000 token prompt + 500 token response):
  Cost = (1000 × $3/1M) + (500 × $15/1M)
       = $0.003 + $0.0075
       = ~$0.01 per call
```

Long system prompts + long conversations = expensive. Keep prompts tight.

---

## 10. Streaming — Get Tokens as They Generate

```python
with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain ViT"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)  # prints token by token
```

Better UX — user sees response being built rather than waiting.

---

## 11. Vision — Send Images

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": "What objects are in this image?"
            }
        ]
    }]
)
```

---

## 12. Base64 — Why Images Need Encoding

APIs communicate via **text** (HTTP/JSON). Images are **binary** — not text safe.

**Base64 = converts binary → safe text characters.**

```
Takes every 3 bytes → converts to 4 printable characters
Only uses: A-Z, a-z, 0-9, +, /  (64 safe characters)
```

```python
import base64

# Read image as binary
with open("cat.jpg", "rb") as f:
    binary_data = f.read()

print(binary_data[:10])
# b'\xff\xd8\xff\xe0\x00\x10JFIF'  ← raw binary, not text safe

# Convert to base64
b64_data = base64.b64encode(binary_data).decode()
print(b64_data[:50])
# '/9j/4AAQSkZJRgABAQAAAQABAAD/...'  ← safe text, fits in JSON
```

**Why .decode()?**
```python
base64.b64encode(data)          # returns bytes: b'/9j/4AAQ...'
base64.b64encode(data).decode() # returns string: '/9j/4AAQ...'
# API expects string, not bytes
```

**Size cost:** Base64 increases file size ~33% (3 bytes → 4 characters).

**Decoding back:**
```python
binary = base64.b64decode(b64_string)
with open("output.jpg", "wb") as f:
    f.write(binary)
```

**Where Base64 is used:**
```
LLM APIs          → send images in JSON
Email attachments → files as text
HTML images       → <img src="data:image/png;base64,...">
JWT tokens        → header.payload encoded as base64
```

**Analogy:** Base64 = translator that converts binary image into a text alphabet that JSON can safely carry — like translating Morse code into readable English for a telegram.

---

## Summary Table

| Concept | Purpose |
|---------|---------|
| system prompt | Model behavior instructions |
| user role | Human input |
| assistant role | Previous model responses |
| temperature | Control randomness |
| max_tokens | Limit output length |
| streaming | Get tokens as generated |
| structured output | Force JSON response |
| vision | Send image + text together |
| base64 | Convert binary image → safe text for API |

---

## One-line Summary

> LLM API = send text/images as JSON → model generates response token by token → base64 encodes images for safe transport → system prompt controls behavior → structured output forces JSON responses.
