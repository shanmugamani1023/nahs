# Day 25 — Multimodal RAG

## Concepts Covered
- What is Multimodal RAG
- Problem with normal RAG for images
- Two approaches (caption-based vs multi-vector)
- Full pipeline
- Code — Multimodal RAG with ChromaDB + Claude
- Normal RAG vs Multimodal RAG
- CV Engineer use cases

---

## 1. What is Multimodal RAG?

Normal RAG: only retrieves text chunks.

**Multimodal RAG = retrieve both images AND text, then answer using both.**

```
Query: "Show me defects found in batch 42"

Normal RAG:      retrieves text → "Batch 42 had surface cracks..."
Multimodal RAG:  retrieves text + images → text description + actual defect photos
```

**Analogy:** Normal RAG = librarian who only gives you text books. Multimodal RAG = librarian who gives you text books + photo albums + diagrams — whatever is most relevant.

---

## 2. The Problem with Normal RAG for Images

```
Normal RAG pipeline:
  Document → chunk text → embed text → store → retrieve text

Images:
  Can't chunk an image into text
  Can't directly embed an image with text embedding model
  Can't store image in text vector DB
```

Solution: **Convert images to text (captions) first, then embed.**

---

## 3. Two Approaches

### Approach 1 — Caption-based (Simple)

```
Image → generate caption → embed caption → store caption + image path
Query → retrieve caption → load original image → send image + caption to LLM
```

```
"product_defect_001.jpg" → "Surface crack on top-left corner, 2mm width"
                                         ↓
                               embed this text → store in vector DB
                               also store image path alongside
```

### Approach 2 — Multi-vector (Advanced)

```
Image → CLIP image encoder → image vector → store in DB
Text  → text encoder       → text vector  → store in DB

Query → encode query → search both DBs → retrieve matching images + text
```

---

## 4. Full Pipeline — Caption-based

```
INDEXING:
  Images → LLaVA/Claude → captions
  PDFs   → extract text
  All text → chunk → embed → store in ChromaDB
  Store image paths alongside captions

QUERYING:
  User query → embed → search ChromaDB
  Retrieve: text chunks + captions (with image paths)
  Load original images from paths
  Send: query + text + images → multimodal LLM → answer
```

---

## 5. Full Code — Multimodal RAG

```python
import anthropic
import chromadb
import base64
import os
from sentence_transformers import SentenceTransformer

# ── Setup ─────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("multimodal_docs")
claude = anthropic.Anthropic(api_key="your-key")

# ── Step 1: Generate Captions for Images ──────────────────
def caption_image(image_path: str) -> str:
    """Use Claude to caption an image"""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail. Include any defects, objects, or important features."
                }
            ]
        }]
    )
    return response.content[0].text

# ── Step 2: Index Images ──────────────────────────────────
image_folder = "./product_images"

for i, filename in enumerate(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(image_folder, filename)

        caption = caption_image(path)
        embedding = embed_model.encode(caption).tolist()

        collection.add(
            documents=[caption],
            embeddings=[embedding],
            metadatas=[{"image_path": path, "filename": filename}],
            ids=[f"img_{i}"]
        )
        print(f"Indexed: {filename} → {caption[:50]}...")

# ── Step 3: Index Text Documents ─────────────────────────
text_docs = [
    "Batch 42 quality report: 3 defects found on 2026-08-20",
    "Defect types: surface crack, discoloration, dent",
    "Standard: surface cracks above 1mm must be rejected",
]
for i, doc in enumerate(text_docs):
    embedding = embed_model.encode(doc).tolist()
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        metadatas=[{"type": "text"}],
        ids=[f"txt_{i}"]
    )

# ── Step 4: Query ─────────────────────────────────────────
def multimodal_rag_query(question: str) -> str:
    query_embedding = embed_model.encode(question).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    content = []

    # Add retrieved images
    for metadata in results["metadatas"][0]:
        if "image_path" in metadata:
            with open(metadata["image_path"], "rb") as f:
                img_data = base64.standard_b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}
            })

    # Add retrieved text + question
    retrieved_text = "\n".join(results["documents"][0])
    content.append({
        "type": "text",
        "text": f"Context:\n{retrieved_text}\n\nQuestion: {question}"
    })

    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

# ── Run ───────────────────────────────────────────────────
answer = multimodal_rag_query("What defects are visible in the product images?")
print(answer)
```

---

## 6. Normal RAG vs Multimodal RAG

| | Normal RAG | Multimodal RAG |
|--|-----------|----------------|
| Input | Text only | Text + Images |
| Retrieval | Text chunks | Text chunks + Image captions |
| LLM input | Text context | Text + actual images |
| Use case | Document Q&A | Visual inspection, reports |
| Complexity | Low | Medium |

---

## 7. CV Engineer Use Cases

```
1. Factory inspection system
   Index: defect images + inspection reports
   Query: "Show similar defects to this one" → retrieves matching images + reports

2. Product manual Q&A
   Index: product images + text manuals
   Query: "How do I fix this error?" → retrieves relevant diagram + instructions

3. Visual audit system
   Index: daily inspection photos + shift reports
   Query: "Any cracks in batch 42?" → retrieves relevant photos + text findings
```

---

## Summary

```
Multimodal RAG  = retrieve images + text, answer using both
Problem         = images can't be embedded directly as text
Solution        = caption images first → embed captions → store with image path

Pipeline:
  Images → caption (Claude) → embed → ChromaDB (with image path in metadata)
  Text   → chunk → embed → ChromaDB
  Query  → retrieve captions + text → load images → send all to LLM → answer

Best for:
  Visual inspection systems
  Document + image Q&A
  Factory/manufacturing AI tools
```
