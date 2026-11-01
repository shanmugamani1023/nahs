# Revision Plan: CV → LLM
**Profile:** Strong practical CV | Need theory depth | Then LLM from ground up  
**Format:** Theory + Code each day | 1 hour/day  
**Total:** ~7 weeks

---

## Part 1: Computer Vision Revision (3 Weeks / 21 Days)

### Week 1: Classical CV — The Theory Behind What You Do

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 1 | Image as Math | Convolution derivation, kernel math | Implement 2D convolution from scratch with NumPy |
| 2 | Color & Histograms | Color space math (RGB↔HSV formulas), histogram equalization | Manual histogram equalization without cv2.equalizeHist |
| 3 | Filtering Theory | Gaussian kernel derivation, frequency domain, Fourier basics | Compare Gaussian vs median on noisy image |
| 4 | Edge Detection | Gradient math, Canny pipeline internals (NMS, hysteresis) | Implement Sobel + NMS manually |
| 5 | Morphology | Structuring element math, opening/closing theory | Build a document cleaner using morphology |
| 6 | Feature Descriptors | HOG math (cell histograms, block normalization), SIFT scale-space | Compute HOG descriptor manually on a patch |
| 7 | Review + Mini Project | — | Image preprocessing pipeline: noise removal → edge → mask |

### Week 2: Deep Learning CV — Architecture Theory

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 8 | CNN Internals | Receptive field, parameter sharing, why conv works for images | Calculate receptive field for a custom CNN |
| 9 | Backprop through Conv | Gradient flow through conv layers, chain rule for filters | Implement forward + backward for a conv layer |
| 10 | Pooling & Normalization | MaxPool gradient, BatchNorm math (mean/var/scale/shift) | BatchNorm from scratch |
| 11 | ResNet Skip Connections | Why gradients vanish, how skip connections fix it, ResNet math | Build ResNet block from scratch in PyTorch |
| 12 | Detection Heads | Anchor boxes math, IoU formula, NMS algorithm | Implement IoU + NMS from scratch |
| 13 | Segmentation Theory | FCN → U-Net architecture, skip connections in U-Net | Build U-Net encoder-decoder from scratch |
| 14 | Review + Mini Project | — | Object detector: custom CNN + detection head |

### Week 3: Modern CV — Transformers & Foundation Models

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 15 | Attention Mechanism | Self-attention math: Q, K, V, softmax(QKᵀ/√d)V | Implement self-attention from scratch |
| 16 | Vision Transformer (ViT) | Patch embedding, positional encoding, CLS token | Build a mini ViT from scratch |
| 17 | CLIP | Contrastive loss, image-text alignment, zero-shot | Use CLIP for zero-shot image classification |
| 18 | DINO / DINOv2 | Self-supervised learning, pseudo-labels, knowledge distillation | Use DINOv2 features for image similarity |
| 19 | SAM (Segment Anything) | Architecture: image encoder + prompt encoder + mask decoder | Use SAM with point/box prompts |
| 20 | Detection: YOLO Theory | YOLOv8 architecture internals, anchor-free heads, loss | Profile YOLOv8 inference, visualize feature maps |
| 21 | Review + Project | — | Zero-shot object detector using CLIP + SAM |

---

## Part 2: LLM (4 Weeks / 28 Days)

### Week 4: Transformer Architecture — The Foundation

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 22 | Attention Deep Dive | Multi-head attention math, why multiple heads | Implement multi-head attention from scratch |
| 23 | Transformer Block | Layer norm, FFN, residual connections, full block | Build a transformer block from scratch |
| 24 | Positional Encoding | Sinusoidal vs learned, RoPE (Rotary PE) used in Llama | Visualize positional encodings |
| 25 | Tokenization | BPE algorithm, WordPiece, SentencePiece, vocab | Implement basic BPE tokenizer |
| 26 | LLM Pretraining | Next token prediction, causal masking, scaling laws | Train a tiny GPT on toy dataset |
| 27 | RLHF & Alignment | SFT → Reward model → PPO, DPO alternative | Compare base vs instruction-tuned model outputs |
| 28 | Review + Project | — | Build a tiny GPT character-level language model |

### Week 5: LLM APIs + Prompt Engineering + RAG

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 29 | LLM APIs | Tokens, context window, temperature, top-p, top-k | Call OpenAI/Claude API, experiment with params |
| 30 | Prompt Engineering | Zero-shot, few-shot, chain-of-thought, structured output | Build a structured JSON extractor |
| 31 | Embeddings | Embedding space, cosine similarity, sentence transformers | Semantic search with sentence-transformers |
| 32 | Vector Databases | FAISS index types (Flat, IVF, HNSW), ChromaDB | Build FAISS index, add + query documents |
| 33 | RAG Pipeline | Chunking strategies, retrieval, reranking | Full RAG: PDF → chunk → embed → retrieve → generate |
| 34 | RAG Evaluation | Faithfulness, relevance, recall (RAGAs) | Evaluate RAG pipeline with RAGAs |
| 35 | Review + Project | — | Domain Q&A bot using RAG over CV research papers |

### Week 6: LangChain + Agents

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 36 | LangChain Core | Chains, LCEL, prompts, output parsers | Build a chain with LCEL |
| 37 | LangChain RAG | Document loaders, text splitters, retrievers | Multi-format RAG (PDF + web + CSV) |
| 38 | Agents & Tools | ReAct loop, tool calling, function calling | Build an agent with web search + calculator tools |
| 39 | LangGraph | State machines, nodes, edges, conditional routing | Build a multi-step agent with LangGraph |
| 40 | Memory Systems | ConversationBuffer, Summary, VectorStore memory | Chatbot with long-term memory |
| 41 | Multi-Agent | CrewAI roles, AutoGen conversation patterns | Build a 2-agent researcher + writer pipeline |
| 42 | Review + Project | — | Autonomous CV paper researcher agent |

### Week 7: Fine-tuning + Multimodal LLMs

| Day | Topic | Theory Focus | Code Practice |
|-----|-------|-------------|---------------|
| 43 | Fine-tuning Theory | When to fine-tune vs RAG, data requirements | Prepare a fine-tuning dataset (instruction format) |
| 44 | LoRA Theory | Low-rank decomposition math, why it works | Fine-tune with LoRA using PEFT library |
| 45 | QLoRA | 4-bit quantization + LoRA, memory math | Fine-tune Llama 3 / Mistral on Colab |
| 46 | Evaluation | Perplexity, BLEU, task-specific evals | Evaluate fine-tuned vs base model |
| 47 | Multimodal LLMs | VLM architecture: vision encoder + LLM, projector | Use LLaVA / Qwen-VL for image Q&A |
| 48 | Multimodal RAG | Multi-vector retrieval, image + text stores | Build multimodal RAG (images + documents) |
| 49 | Final Project | — | **Capstone: Visual AI Inspector** — image in → YOLO detects → LLaVA describes → RAG adds context → Claude writes report |

---

## Progress Tracker

| Day | Date | Topic | Status | Notes |
|-----|------|-------|--------|-------|
| 1 | 2026-07-31 | Image as Math — Convolution from scratch | ✅ | [notes](notes/day01_convolution.md) |
| 2 | 2026-08-02 | Color Spaces & Histograms | ✅ | [notes](notes/day02_color_histograms.md) |
| 3 | 2026-08-03 | Filtering Theory — Gaussian, Frequency Domain | ✅ | [notes](notes/day03_filtering_theory.md) |
| 4 | 2026-08-03 | Edge Detection — Sobel, Canny Pipeline | ✅ | [notes](notes/day04_edge_detection.md) |
| 5 | 2026-08-04 | Morphological Operations | ✅ | [notes](notes/day05_morphology.md) |
| 6 | 2026-08-05 | Geometric Transforms — Affine, Perspective, Homography | ✅ | [notes](notes/day06_geometric_transforms.md) |
| 7 | 2026-08-06 | Feature Detection — HOG, SIFT, ORB, Matching | ✅ | [notes](notes/day07_feature_detection.md) |
| 8 | 2026-08-11 | CNN Internals — Local connectivity, receptive field, backprop, ResNet | ✅ | [notes](notes/day08_cnn_internals.md) |
| 9 | 2026-08-11 | Object Detection — Anchor boxes, YOLO pipeline, training, NMS | ✅ | [notes](notes/day09_object_detection.md) |
| 10 | 2026-08-11 | Segmentation — Semantic/Instance/Panoptic, U-Net, Dice loss | ✅ | [notes](notes/day10_segmentation.md) |
| 11 | 2026-08-12 | Vision Transformer — Patches, Self-attention, CLS token, Task-wise usage | ✅ | [notes](notes/day11_vit.md) |
| 12 | 2026-08-12 | CLIP — Contrastive learning, dual encoders, zero-shot classification | ✅ | [notes](notes/day12_clip.md) |
| 13 | 2026-08-13 | SAM — Prompt-based segmentation, ViT-H encoder, mask decoder, zero-shot | ✅ | [notes](notes/day13_sam.md) |
| 14 | 2026-08-14 | DINOv2 — Self-supervised, student-teacher, segmentation, depth estimation | ✅ | [notes](notes/day14_dinov2.md) |
| 15 | 2026-08-14 | Attention Mechanism — Q, K, V, scaled dot-product, multi-head attention | ✅ | [notes](notes/day15_attention.md) |
| 16 | 2026-08-15 | Transformer Architecture — Encoder, decoder, FFN, masked attention, cross-attention | ✅ | [notes](notes/day16_transformer.md) |
| 17 | 2026-08-15 | LLM Basics — Tokenization, next token prediction, temperature, sampling | ✅ | [notes](notes/day17_llm_basics.md) |
| 18 | 2026-08-15 | LLM APIs — Claude/OpenAI API, prompt engineering, vision, base64 | ✅ | [notes](notes/day18_llm_apis.md) |
| 19 | 2026-08-18 | RAG — Chunking, embeddings, vector DB, retrieval, ChromaDB + Claude | ✅ | [notes](notes/day19_rag.md) |
| 20 | 2026-08-19 | LangChain — LCEL, document loaders, RAG pipeline, memory | ✅ | [notes](notes/day20_langchain.md) |
| 21 | 2026-08-19 | Fine-tuning + LoRA + QLoRA — parallel layers, low rank, 4-bit quantization | ✅ | [notes](notes/day21_finetuning_lora.md) |
| 22 | 2026-08-20 | Agents + Tool Calling — ReAct loop, reasoning, tools, LangChain agent | ✅ | [notes](notes/day22_agents.md) |
| 23 | 2026-08-20 | LangGraph — nodes, edges, state, conditional branching, multi-agent | ✅ | [notes](notes/day23_langgraph.md) |
| 24 | 2026-08-20 | Multimodal LLMs — LLaVA architecture, projector, YOLO+LLaVA pipeline | ✅ | [notes](notes/day24_multimodal_llms.md) |
| 25 | 2026-08-21 | Multimodal RAG — caption-based, multi-vector, ChromaDB + Claude pipeline | ✅ | [notes](notes/day25_multimodal_rag.md) |
| 26 | 2026-08-28 | Project Explanation — Retail AI Chatbot architecture, interview Q&A, scaling, security | ✅ | [notes](notes/project_retail_chatbot.md) |

---

## How Each Day Works (1 hour)

```
[0:00–0:15]  Theory — read concept, understand the math
[0:15–0:45]  Code — implement it from scratch or use it
[0:45–1:00]  Note one key insight in this file
```
