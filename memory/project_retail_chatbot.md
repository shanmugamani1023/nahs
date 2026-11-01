---
name: project-retail-chatbot
description: "Retail AI Chatbot architecture, tools, tech stack, key numbers"
metadata:
  node_type: memory
  type: project
  originSessionId: 1a78bc16-e4b9-42f2-9af8-f8b937107c6c
  modified: 2026-11-01T08:43:38.560Z
---

## Architecture
Mobile App → API Gateway (60 RPM rate limit) → FastAPI (4 workers, async) → LangGraph Agent → Tools → Response

## Input Pipeline
Pydantic validation (500 char limit) → Presidio PII masking → Rebuff injection check

## LangGraph Agent
- Pattern: ReAct (Reason → Act → Observe), max_iterations=10
- 6 Tools: search_products (Qdrant), check_stock (PostgreSQL), get_transaction (PostgreSQL), create_ticket (Freshdesk+PostgreSQL), check_ticket_status (Freshdesk), get_store_info (PostgreSQL)
- Barcode is the link between Qdrant and PostgreSQL (not product_id)

## LLM Stack
- Tier 1: Ollama llama3.2 8B on Azure NC4as T4 v3 (16GB VRAM, 50-55 tokens/sec)
- Tier 2: Groq API (llama3.3 70B) — different provider for redundancy
- Tier 3: Gemini Flash — last resort
- Temperature: 0.1 (factual retail chatbot)
- max_new_tokens: 500

## Qdrant
- 300 products, 384 dimensions (all-MiniLM-L6-v2)
- score_threshold=0.7, HNSW index
- Stores: name, barcode, features, description, tags
- ~1.5MB total size

## Redis
- Last 10 messages per user, TTL=1 hour, key=history:{user_id}

## Evaluation
- 1000 test cases with ground truth
- LLM-as-judge scoring
- 90% → 95% accuracy (hallucination fixed with system prompt hardening)
- LangSmith for production monitoring

## HITL
- Complaint tickets go to Freshdesk
- Human team reviews and resolves — agent never auto-refunds
- 6 complaint types: Missing Product, Wrong Product Billed, Wrong Quantity, Wrong Bill Amount, Damaged Product, Expired Product

## Key Numbers
- 300 products, 384 dims, score_threshold=0.7
- Last 10 messages, TTL=1 hour, max_iterations=10
- 60 RPM, 4 workers, ~100 concurrent users
- T4 16GB VRAM, 50-55 tokens/sec, ~$380/month Azure

**Why:** Production system built at Digit7 for autonomous retail store customer support.
**How to apply:** Use these numbers confidently in interviews — all validated.
