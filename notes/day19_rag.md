# Day 19 — RAG (Retrieval Augmented Generation)

## Concepts Covered
- Problem RAG solves
- Full RAG pipeline
- Chunking strategies
- Embedding models
- Vector databases
- Retrieval + cosine similarity
- Prompt construction
- RAG vs Fine-tuning
- Code example (ChromaDB + Claude)

---

## 1. The Problem

LLM knows everything from training data — but nothing about YOUR documents.

```
You: "What does our company policy say about remote work?"
LLM: "I don't have access to your company documents..."
```

LLM has a **knowledge cutoff** and no access to private/custom data.

**RAG solution:** Fetch relevant documents → give to LLM as context → LLM answers from context.

**Analogy:** Open-book exam. Instead of memorizing everything, you retrieve the right pages from your textbook and read the answer from there.

---

## 2. Full RAG Pipeline

```
INDEXING (once):
  Documents → Chunk → Embed → Store in Vector DB

QUERYING (every request):
  Question → Embed → Search Vector DB → Top-K chunks → Prompt → LLM → Answer
```

### Step by Step:

```
1. Load documents (PDF, web, CSV...)
2. Split into chunks (256–512 tokens each)
3. Embed each chunk → 768-dim vector
4. Store vectors in vector DB

5. User asks a question
6. Embed question → 768-dim vector
7. Cosine similarity: question vector vs all chunk vectors
8. Retrieve top-K most similar chunks
9. Build prompt: "Use this context to answer: {chunks} \n Question: {question}"
10. LLM generates answer from context
```

---

## 3. Chunking — Splitting Documents

Too large → irrelevant content included  
Too small → not enough context

```
Chunk size:     256–512 tokens  (sweet spot)
Chunk overlap:  50 tokens       (preserve context at boundaries)

"...end of chunk 1..." | "...overlap...start of chunk 2..."
```

**Why overlap?** Key sentences might be split across chunk boundary. Overlap prevents losing that context.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(document_text)
```

---

## 4. Embedding — Text to Vector

Convert text → fixed-size vector that captures meaning.

```
"The cat is on the mat"  → [0.2, 0.8, -0.1, 0.5, ...]  768-dim
"A feline sits on a rug" → [0.21, 0.79, -0.09, 0.51, ...]  768-dim
(very similar! even different words)

"Stock market crashes"   → [-0.5, 0.1, 0.9, -0.3, ...]  768-dim
(very different direction)
```

**Popular embedding models:**

| Model | Dim | Speed | Use case |
|-------|-----|-------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Local, free |
| text-embedding-3-small | 1536 | Fast | OpenAI API |
| text-embedding-3-large | 3072 | Slower | Best quality |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is YOLO?")
print(embedding.shape)  # (384,)
```

---

## 5. Vector Database — Store and Search

Stores embeddings + original text. Fast similarity search.

```
Store: chunk_text + embedding → DB
Query: question_embedding → cosine similarity → top-K results
```

**Popular vector DBs:**

| DB | Type | Best For |
|----|------|----------|
| FAISS | Local (in-memory) | Fast prototyping, no setup |
| ChromaDB | Local (persistent) | Small-medium projects |
| Pinecone | Cloud | Production scale |
| Weaviate | Cloud/self-hosted | Enterprise |

**Cosine similarity:**
```
similarity = (A · B) / (|A| × |B|)

Range: -1 (opposite) to 1 (identical)
0.9+ = very similar
0.5- = not related
```

**Analogy:** Every chunk is a book on a shelf. Each book has a "topic fingerprint" (vector). Your question also has a fingerprint. Find the 3 books whose fingerprints are closest to your question.

---

## 6. Prompt Construction — Key Part

```python
context = "\n".join(retrieved_chunks)

prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
```

**"Only use context"** — prevents hallucination. LLM cannot use its training knowledge, only retrieved chunks.

---

## 7. Full Code Example — ChromaDB + Claude

```python
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

# ── 1. Setup ──────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("my_docs")
claude_client = anthropic.Anthropic(api_key="your-key")

# ── 2. Index Documents ────────────────────────────────────
documents = [
    "YOLO is a real-time object detection algorithm.",
    "ViT splits an image into 16×16 patches and applies self-attention.",
    "SAM is a prompt-based segmentation model from Meta.",
    "CLIP aligns image and text embeddings in the same space.",
    "DINOv2 uses self-supervised student-teacher training.",
]

embeddings = embed_model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# ── 3. Query ──────────────────────────────────────────────
question = "How does ViT process an image?"
question_embedding = embed_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved = results["documents"][0]

# ── 4. Prompt + LLM ───────────────────────────────────────
context = "\n".join(retrieved)
prompt = f"""Answer using ONLY this context:

{context}

Question: {question}"""

response = claude_client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}]
)

print(response.content[0].text)
# "ViT splits an image into 16×16 patches and applies self-attention."
```

---

## 8. RAG vs Fine-tuning

| | RAG | Fine-tuning |
|--|-----|-------------|
| When | New/changing data | Fixed domain knowledge |
| Data needed | Any documents | 100s–1000s labeled examples |
| Cost | Low (just embeddings) | High (GPU training) |
| Update | Just re-index | Re-train |
| Hallucination | Low (grounded in docs) | Higher risk |
| Latency | +100ms retrieval | Same as base model |
| Best for | Company docs, Q&A | Tone, style, specialized tasks |

**Rule of thumb:**
```
Data changes often?     → RAG
Need custom style/tone? → Fine-tune
Need domain knowledge?  → RAG
Have lots of labeled data + fixed task? → Fine-tune
```

---

## Summary Table

| Step | What Happens | Key Tool |
|------|-------------|---------|
| Chunk | Split doc into ~500 token pieces | RecursiveCharacterTextSplitter |
| Embed | Convert chunk → vector | SentenceTransformer / OpenAI |
| Store | Save vector + text | ChromaDB / FAISS / Pinecone |
| Query embed | Convert question → vector | Same embedding model |
| Retrieve | Cosine similarity → top-K chunks | Vector DB query |
| Prompt | Inject chunks into prompt | Manual template |
| Generate | LLM answers from context | Claude / GPT |

---

## One-line Summary

> RAG = chunk documents → embed → store in vector DB → at query time, retrieve similar chunks → inject as context into LLM prompt → LLM answers from context, not hallucination.
