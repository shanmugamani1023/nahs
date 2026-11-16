# AI / ML Engineering — Mentor Plan

**Started:** 2026-11-16
**Pace:** 1–1.5 hrs/day + 1 DSA problem daily
**Goal:** Go from "I can build it" to "I can explain why it works"

---

## Where I Stand

| Layer | What it covers | Level |
|---|---|---|
| 1. Foundations | Python, DSA, ML math (linear algebra, probability, gradients) | Weak — fix first |
| 2. ML & DL theory | Why models learn, overfit, fail | Weak — fix first |
| 3. LLM systems | Transformers, RAG, agents, fine-tuning, eval | Medium |
| 4. Production | APIs, deployment, monitoring, cost, latency, safety | Strong |

---

## The Path (~16 weeks)

| Phase | Weeks | Learn | Build |
|---|---|---|---|
| **0. Foundations** | ongoing | 1 DSA problem/day, math only as needed | `dsa/` grows daily |
| **1. Classical ML** | 2 | Bias–variance, regularization, metrics, cross-validation, trees/XGBoost | Tabular model with honest evaluation |
| **2. Deep learning from scratch** | 2 | Backprop by hand, optimizers, init, normalization | micrograd-style autograd + MLP in pure NumPy |
| **3. Transformers** | 2 | Attention, tokenization, positional encoding, GPT architecture | Char-level mini-GPT from scratch |
| **4. LLM applications** | 3 | Structured outputs, tool calling, RAG done right (chunking, hybrid search, reranking), agents | Rebuild retail chatbot core — understand every line |
| **5. Fine-tuning** | 2 | LoRA/QLoRA, fine-tune vs RAG, data quality | Fine-tune a small model on a retail task |
| **6. Production & LLMOps** | 2 | Eval (RAGAS, LLM-as-judge), observability, guardrails, cost/latency | Dockerized, evaluated API |
| **7. System design & interviews** | 2 | ML/LLM system design, mock interviews | Capstone + 2 polished project write-ups |

---

## Free Courses

> Rule: one main course per phase. Finish it before adding another.

### DSA (daily)
- **[NeetCode 150](https://neetcode.io)** — pattern-wise, free video for each problem. Go in order.
- Backup: **Striver's A2Z DSA Sheet** (takeuforward) — more practice per topic.

### Math (alongside, as needed)
- **3Blue1Brown** — *Essence of Linear Algebra*, *Essence of Calculus*, *Neural Networks* series.
- **StatQuest (Josh Starmer)** — stats + classical ML, great for "explain in 60 seconds."

### By Phase

| Phase | Main course | Why |
|---|---|---|
| 1. Classical ML | **[Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)** + **StatQuest** per algorithm | Short, practical, covers interview theory |
| 2–3. DL + Transformers | **Andrej Karpathy — *Neural Networks: Zero to Hero*** ([karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)) | **Core course.** micrograd → makemore → GPT from scratch → tokenizer |
| 3. Transformers (reading) | **Jay Alammar — *The Illustrated Transformer*** + **[Hugging Face LLM Course](https://huggingface.co/learn)** | Visual intuition, then HF hands-on |
| 4. LLM apps & agents | **DeepLearning.AI short courses** (RAG, LangGraph, evaluation) + **HF Agents Course** + **[Anthropic courses](https://github.com/anthropics/courses)** | Short and current — fill gaps, not main track |
| 5. Fine-tuning | **HF LLM Course** fine-tuning chapters + **Unsloth free Colab notebooks** | LoRA/QLoRA on a free GPU |
| 6. Production | **[Made With ML](https://madewithml.com)** + **DataTalksClub LLM Zoomcamp / MLOps Zoomcamp** (GitHub) | End-to-end MLOps with real projects |

*If a link has moved, search the course name.*

---

## How We Work

- **I write the code first.** Mentor reviews, asks "why", shows a version only after I've tried.
- **Every topic ends with:** explain it to an interviewer in 60 seconds.
- **Commit daily.**

---

## This Week

1. **Daily DSA:** one NeetCode problem — start with `two_sum` in [dsa/02_hashmap.py](../dsa/02_hashmap.py) (O(n), no looking up).
2. **Main track:** Karpathy video 1 — *"The spelled-out intro to neural networks and backpropagation: building micrograd."* Code along in `dl/01_micrograd.py`. Pause and type, don't copy.
3. **Side:** 3Blue1Brown *Neural Networks* chapters 1–4.

### Diagnostic (answer in my own words)
1. Why does adding L2 regularization reduce overfitting?
2. In backprop, what exactly is the gradient of the loss w.r.t. a weight telling you?
3. Why does attention divide by √d_k?
4. My RAG returns irrelevant chunks — list 3 things I'd check, in order.
5. Precision vs recall — which matters more for the billing pipeline's product classifier, and why?

---

## Progress Log

| Date | DSA | Main track | Notes |
|---|---|---|---|
| 2026-11-16 | — | Plan created | |
