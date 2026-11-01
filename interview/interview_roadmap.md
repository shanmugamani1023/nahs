# AI/ML Engineer Interview Roadmap
## Complete Guide — A to Z

---

## Your Current Strength Assessment

```
STRONG ✓ (Already have):
  Computer Vision     → Product Handler, Pose Estimator, Depth Estimator
  RAG + LangChain     → Retail AI Chatbot
  Production systems  → Edge deployment, Redis, Azure, RTSP
  Python + OpenCV     → Strong hands-on
  Real metrics        → 1.5s cloud vs 500ms edge, 95% accuracy, 30+ cameras

GAPS to fill:
  ML Theory           → Fundamentals interviewers always ask
  LLM/GenAI depth     → Key for AI Engineer role
  System Design       → Senior level questions
  DSA                 → Screening round filter
  NLP basics          → Supporting knowledge
```

---

## Your Biggest Advantage

```
Most candidates:
  → Know theory, no production experience

You:
  → REAL production CV systems deployed in actual retail store
  → Edge deployment (Jetson), Redis, Azure, RTSP cameras
  → Built RAG chatbot end to end
  → HITL pipelines, confidence thresholds, fallback systems
  → Real numbers: 1.5s→500ms, 95% accuracy, 30+ cameras, ₹24k/month cost

This is RARE for 4 years experience.
Use it aggressively in every interview.
```

---

# PHASE 1: ML Fundamentals
## (Week 1-2) — Always Asked

---

### 1. Types of Machine Learning
```
Supervised:
  → Learns from labeled data (input + correct output)
  → Examples: classification, regression
  → Your work: ResNet classifier (product images + labels)

Unsupervised:
  → Learns from unlabeled data (finds patterns)
  → Examples: clustering, dimensionality reduction
  → Example: customer segmentation

Reinforcement Learning:
  → Agent learns by trial and error (reward/penalty)
  → Examples: game playing, robotics
  → Not commonly used in retail CV
```

### 2. Bias vs Variance
```
Bias:
  → Model too simple → misses patterns → underfitting
  → High training error + high test error
  → Fix: bigger model, more features

Variance:
  → Model too complex → memorizes training data → overfitting
  → Low training error + high test error
  → Fix: more data, dropout, regularization

Sweet spot: low bias + low variance = good model
```

### 3. Overfitting vs Underfitting
```
Overfitting:
  → Training accuracy: 99%
  → Validation accuracy: 70%
  → Model memorized training data
  → Fix: more data, augmentation, dropout, early stopping

Underfitting:
  → Training accuracy: 70%
  → Validation accuracy: 70%
  → Model too simple
  → Fix: bigger model, more epochs, more features
```

### 4. Train / Validation / Test Split
```
Training set (70%):   model learns from this
Validation set (15%): tune hyperparameters, check overfitting
Test set (15%):       final evaluation — touch only once

Rule: NEVER use test set during training or tuning
      Test set = real world simulation

Your project: 300 images per product
  → Train: 210, Val: 45, Test: 45
```

### 5. Loss Functions
```
MSE (Mean Squared Error):
  → Regression problems (predict price, temperature)
  → Penalizes large errors more

Cross Entropy:
  → Classification problems
  → Binary (2 classes) or Categorical (multiple classes)
  → Your ResNet classifier uses categorical cross entropy

Focal Loss:
  → Imbalanced datasets (rare products vs common products)
  → Down-weights easy examples → focuses on hard ones
  → Good for your product classifier (rare SKUs)
```

### 6. Optimizers
```
SGD (Stochastic Gradient Descent):
  → Basic, reliable, needs careful learning rate tuning
  → Momentum variant is better

Adam:
  → Adaptive learning rate per parameter
  → Works well out of the box
  → Most commonly used

AdamW:
  → Adam + weight decay (better regularization)
  → Preferred for transformer models (ViT, LLM fine-tuning)

Learning Rate:
  → Too high → overshoots, doesn't converge
  → Too low → too slow, gets stuck
  → Typical: 1e-3 for Adam, 1e-4 for fine-tuning
```

### 7. Evaluation Metrics
```
Accuracy:
  → Correct predictions / Total predictions
  → Misleading for imbalanced data

Precision:
  → Of all predicted positives, how many are actually positive?
  → Use when false positive is costly
  → Example: spam detection (don't mark real emails as spam)

Recall:
  → Of all actual positives, how many did we catch?
  → Use when false negative is costly
  → Example: cancer detection (don't miss actual cancer)

F1 Score:
  → Harmonic mean of precision and recall
  → Use when both matter
  → Your product classifier: F1 per class

AUC-ROC:
  → How well model separates classes
  → 1.0 = perfect, 0.5 = random guess
  → Use for binary classification

Your project target: 95% accuracy per class
```

### 8. Handling Class Imbalance
```
Problem:
  Common product (Pepsi): 5000 samples
  Rare product (imported item): 50 samples
  → Model biased toward Pepsi

Solutions:
  1. Class weights: rare class gets higher loss weight
  2. Oversample: augment rare class images
  3. Undersample: reduce common class samples
  4. Focal loss: automatically handles imbalance
  5. Collect more data for rare classes
```

---

# PHASE 2: Deep Learning
## (Week 2-3)

---

### 1. How CNN Works
```
Convolution layer:
  → Slide small filter (3×3) over image
  → Compute dot product at each position
  → Output: feature map (where is this pattern?)
  → Multiple filters → multiple feature maps

Pooling layer:
  → Reduce spatial size (2×2 max pool → halves dimensions)
  → Keep most important features
  → Reduces computation

ReLU activation:
  → Remove negative values (set to 0)
  → Adds non-linearity

Fully connected layer:
  → Flatten → classify into N classes
```

### 2. Backpropagation
```
Forward pass:  input → layers → prediction → loss calculated
Backward pass: loss → gradient flows backward → weights updated

Chain rule:    gradient of each layer = gradient from next layer × local gradient
Optimizer:     uses gradients to update weights (Adam, SGD)

Why vanishing gradient?
  → Deep networks → gradients become very small → early layers don't learn
  → Fix: ResNet skip connections, batch norm, ReLU
```

### 3. Transfer Learning
```
Pretrained model (ImageNet):
  → Already learned: edges, textures, shapes, objects
  → 1000 classes, 1.2M images, months of training

Fine-tuning approach:
  Option 1 — Feature extraction:
    → Freeze all layers → only train final FC layer
    → Use when: small dataset, similar domain
    → Your product classifier (300 images/class)

  Option 2 — Full fine-tuning:
    → Unfreeze all layers → train everything
    → Use when: large dataset, different domain
    → Slower but better accuracy

Code:
  model = WideResNet50(pretrained=True)
  model.fc = nn.Linear(2048, num_products)  # replace last layer
```

### 4. Key Architectures
```
VGG16/19:
  → Deep network, only 3×3 convolutions
  → Simple but large (138M params)
  → Good baseline

ResNet50:
  → Skip connections → solve vanishing gradient
  → Can train 100+ layers
  → Most widely used backbone

WideResNet50:
  → Wider channels instead of more depth
  → Better accuracy than ResNet50
  → Used in your Product Handler

EfficientNet:
  → Scale width + depth + resolution together
  → Best accuracy/parameter ratio
  → Good for production

MobileNet:
  → Depthwise separable convolutions → 8x fewer ops
  → For edge devices, mobile
  → Slightly less accurate but much faster
```

### 5. Attention Mechanism
```
Problem with CNN:
  → Only sees local region (kernel size)
  → Can't connect distant parts of image

Attention:
  → Every position can attend to every other position
  → "Which parts of image are most relevant?"
  → Score = how much to focus on each part

Self-attention (Transformer):
  → Query × Key → attention score → weighted sum of Values
  → Captures global context

Used in: ViT, YOLOX, SAM, all LLMs
```

---

# PHASE 3: LLM + GenAI
## (Week 3-4) — Most Important for AI Engineer

---

### 1. How LLM Works
```
Training:
  → Trained on billions of text documents
  → Learns to predict next token
  → Parameters encode world knowledge

Inference:
  → Input text → tokenize → vectors → transformer layers → output token
  → Auto-regressive: generates one token at a time
  → Temperature: controls randomness (0=deterministic, 1=creative)

Context window:
  → Max tokens model can see at once
  → GPT-4: 128k, Claude: 200k, Gemini: 1M
  → NOT a rate limit — it's memory limit per conversation
```

### 2. Prompt Engineering
```
Zero-shot:
  → "Classify this as positive or negative: 'Great product!'"
  → No examples given

Few-shot:
  → Give 2-3 examples before the actual question
  → Better for complex tasks

Chain of Thought:
  → "Think step by step..."
  → Makes LLM reason before answering
  → Better accuracy on math/logic problems

System prompt:
  → Sets behavior for entire conversation
  → "You are a retail assistant. Only answer product questions."
  → Most important for production chatbots
```

### 3. RAG — Full Pipeline
```
Why RAG:
  → LLM doesn't know YOUR store inventory
  → LLM knowledge has cutoff date
  → LLM hallucinates → RAG grounds in real data

Pipeline:
  1. Data → chunk → embed → store in Qdrant
  2. User query → embed → search Qdrant → top K chunks
  3. Reranker → reorder by relevance (optional but recommended)
  4. Retrieved chunks + query → LLM prompt
  5. LLM generates answer from real data

Key decisions:
  Chunk size:   200-500 tokens (one product = one chunk)
  Overlap:      50 tokens between chunks
  Top K:        3-5 results
  Embedding:    all-MiniLM-L6-v2 (free) or text-embedding-3-small (best)
  Distance:     Cosine similarity for text
```

### 4. Fine-tuning vs RAG
```
RAG:
  → No model training needed
  → Data can be updated anytime (just re-embed)
  → Good for: frequently changing data (inventory)
  → Your chatbot uses RAG ✓

Fine-tuning:
  → Retrain model on your data
  → Model learns your domain deeply
  → Good for: specific writing style, domain language
  → Expensive, needs 1000s of examples
  → Data can't be easily updated

Hybrid:
  → Fine-tune on domain language
  → RAG for real-time data
  → Best quality, most complex
```

### 5. LangChain Components
```
LLM:        ChatGroq, ChatOpenAI, ChatAnthropic
Tools:      @tool decorator → functions agent can call
Agent:      LLM + tools → decides which tool to use
Memory:     ConversationBufferMemory, Redis-backed memory
Chain:      sequence of operations (prompt → LLM → output parser)
Prompt:     ChatPromptTemplate with system + history + human messages
```

### 6. LLM Evaluation
```
How to measure chatbot quality?

Automated metrics:
  → BLEU score (text similarity)
  → BERTScore (semantic similarity)
  → Faithfulness (is answer grounded in retrieved docs?)
  → Relevance (does answer address the question?)

Human evaluation:
  → Rate: helpful? accurate? concise?
  → Gold standard but expensive

LangSmith:
  → Traces every agent step
  → Latency, token count, cost per call
  → Easy debugging when bot gives wrong answer
```

### 7. Hallucination — Causes and Prevention
```
Why LLM hallucinates:
  → Generates plausible-sounding text
  → Doesn't know what it doesn't know
  → Fills gaps with confident-sounding guesses

Prevention:
  → RAG: ground answers in real data
  → System prompt: "Only answer based on provided context"
  → Temperature = 0: deterministic, less creative (less hallucination)
  → Output validation: check if answer contradicts retrieved data
  → Confidence scoring: flag low-confidence answers
```

### 8. Token Economics
```
1 token ≈ 4 characters

Claude Sonnet 3.5:
  Input:  $3 per 1M tokens
  Output: $15 per 1M tokens

Per message cost (your chatbot):
  Input: 2200 tokens × $0.000003 = $0.0066 = ₹0.55
  Output: 200 tokens × $0.000015 = $0.003 = ₹0.25
  Total per message: ~₹0.80

100 users × 10 messages/day:
  = ₹800/day = ₹24,000/month

Cost control:
  → Trim history to last 10 messages
  → Keep output short
  → Use cheaper model for simple queries
  → Cache common answers
```

---

# PHASE 4: System Design for ML
## (Week 4-5) — Senior Role Differentiator

---

### How to Approach System Design

```
Step 1: Clarify requirements (2 min)
  → Scale? (100 users or 1M users?)
  → Real-time or batch?
  → Accuracy vs latency tradeoff?

Step 2: High level design (5 min)
  → Draw main components
  → Data flow

Step 3: Deep dive each component (10 min)
  → Data pipeline
  → Model serving
  → Storage

Step 4: Scale and edge cases (5 min)
  → What if component fails?
  → How to scale?
```

### Common System Design Questions

```
1. Design product recommendation system
   → User history → collaborative filtering / content-based
   → Feature store, embedding model, vector DB
   → A/B test different recommendation strategies

2. Design real-time fraud detection
   → Transaction → feature extraction → model → decision
   → Low latency required (< 100ms)
   → High recall (don't miss fraud) even if more false positives

3. Design image classification pipeline at scale
   → Data collection → labeling → training → evaluation → serving
   → MLflow for experiment tracking
   → Model registry → staging → production

4. Design RAG chatbot (you know this well)
   → Mobile → API Gateway → FastAPI → Redis → Agent → Qdrant + PostgreSQL

5. Design model retraining pipeline
   → Monitor data drift → trigger retraining → evaluate → deploy
   → Automated vs manual approval
```

### ML System Design Components

```
Data Pipeline:
  → Collection → validation → preprocessing → feature store

Training Pipeline:
  → Experiment tracking (MLflow)
  → Hyperparameter tuning
  → Distributed training

Model Registry:
  → Version control for models
  → Staging → production deployment
  → Rollback capability

Serving:
  → REST API (FastAPI)
  → Batch inference vs real-time
  → Load balancing

Monitoring:
  → Data drift (input distribution changed?)
  → Model drift (performance degraded?)
  → Latency, throughput, error rate
```

---

# PHASE 5: MLOps + Production
## (Week 5)

---

### Key MLOps Concepts

```
MLflow:
  → Experiment tracking (log params, metrics, artifacts)
  → Model registry (version, stage: staging/production)
  → Model serving

Docker:
  → Package model + dependencies into container
  → Runs same everywhere (dev, staging, prod)
  → Dockerfile → docker build → docker run

Kubernetes:
  → Orchestrate multiple Docker containers
  → Auto-scaling, load balancing, rolling updates
  → Used for 10k+ users (AKS in Azure)

CI/CD for ML:
  → Code change → auto test → auto deploy
  → Model change → auto evaluate → deploy if better

DVC (Data Version Control):
  → Version control for datasets and models
  → Like git but for large files
```

### Model Optimization for Production

```
Quantization:
  FP32 → FP16: 2x smaller, similar accuracy
  FP32 → INT8: 4x smaller, slight accuracy drop
  Used for: edge deployment, faster inference

Pruning:
  → Remove near-zero weights
  → Smaller, faster model
  → Structured pruning → hardware friendly

TensorRT:
  → NVIDIA's inference optimizer
  → Converts PyTorch → optimized engine
  → 2-5x faster on NVIDIA GPU
  → Used in production CV systems

ONNX:
  → Standard model format
  → Convert PyTorch → ONNX → deploy anywhere
  → torch.onnx.export(model, dummy_input, "model.onnx")
```

### Edge Deployment
```
What it is:
  → Run AI model on local device (not cloud)
  → Jetson, Raspberry Pi, mobile, smart camera

Why edge:
  → Low latency (no network round trip)
  → Data privacy (video stays on device)
  → Works offline
  → One-time hardware cost

Your project:
  → Cloud: 1.5 sec latency (tested)
  → Edge: 500ms latency
  → Customer video never leaves store

Challenges:
  → Limited compute → must optimize model
  → Limited memory → careful management
  → Updates → OTA (over the air) deployment
```

---

# PHASE 6: DSA
## (Daily — 1 Problem/Day)

---

### Week-by-Week Plan

```
Week 1: Arrays + Strings (Done: Two Sum, Best Time to Buy Stock)
  → Sliding Window
  → Kadane's Algorithm (Max subarray)
  → String manipulation

Week 2: Two Pointer + Hashing (Done: Two Sum II, Valid Parentheses)
  → Remove Duplicates (next)
  → Container with most water
  → Group anagrams

Week 3: Stack + Queue
  → Min stack
  → Daily temperatures
  → Evaluate reverse polish notation

Week 4: Linked List
  → Reverse linked list
  → Detect cycle
  → Merge two sorted lists

Week 5: Trees
  → Max depth of binary tree
  → Validate BST
  → Level order traversal (BFS)

Week 6: Dynamic Programming (Easy)
  → Climbing stairs
  → House robber
  → Coin change
```

### Problems Done So Far
```
✓ Two Sum (HashMap)
✓ Best Time to Buy Stock (Greedy)
✓ Valid Parentheses (Stack)
✓ Reverse String (Two Pointer)
✓ Two Sum II (Two Pointer sorted)
→ Next: Remove Duplicates from Sorted Array
```

---

# Interview Question Categories

---

## Theory Questions
```
1. Explain bias-variance tradeoff
2. How does backpropagation work?
3. What is attention mechanism?
4. RAG vs fine-tuning — when to use which?
5. How do you handle class imbalance?
6. What is overfitting? How to prevent?
7. Difference between precision and recall
8. How does transformer work?
9. What is transfer learning?
10. How do you evaluate a RAG system?
```

## Coding Questions
```
1. Implement K-means clustering
2. Implement linear regression from scratch
3. Write a function to compute cosine similarity
4. Implement sliding window maximum
5. Binary search on sorted array
6. DSA: arrays, strings, trees (LeetCode)
```

## PyTorch Questions
```
Basics:
1.  What is a tensor? How is it different from numpy array?
2.  How do you move tensor to GPU?
3.  What is requires_grad? Why is it needed?
4.  What is torch.no_grad()? When to use it?
5.  Difference between .detach() and .no_grad()?

Model Building:
6.  How do you define a custom model in PyTorch?
7.  What is nn.Module? What methods must you implement?
8.  What is forward() method?
9.  How do you freeze layers during fine-tuning?
10. How do you replace the last layer of a pretrained model?

Training Loop:
11. Write a complete PyTorch training loop
12. What is optimizer.zero_grad()? Why is it needed?
13. What is loss.backward()? What does it do?
14. What is optimizer.step()? What does it do?
15. What is the order of: zero_grad, forward, backward, step?

Data:
16. What is Dataset class? What methods must you implement?
17. What is DataLoader? Why use it?
18. What is transforms in torchvision?
19. How do you do data augmentation in PyTorch?

Saving/Loading:
20. How do you save and load a model in PyTorch?
21. Difference between state_dict and full model save?

Debugging:
22. How do you check GPU memory usage?
23. What causes CUDA out of memory error? How to fix?
24. What is gradient clipping? When to use it?
```

## Project Questions (Your Strongest Area)
```
1. Explain Product Handler flow
2. Explain Pose Estimator flow
3. Explain AI Chatbot architecture
4. What problem did you solve? (QC story)
5. Why SAM over YOLO segmentation?
6. How did you handle Xovis failures?
7. How do you scale to 10k users?
8. What would you improve?
```

## System Design Questions
```
1. Design a recommendation system
2. Design a real-time object detection pipeline
3. Design a model retraining pipeline
4. Design a RAG chatbot (practice this)
5. How would you detect data drift?
```

## Behavioral Questions (STAR Format)
```
1. Tell me about yourself (2 min pitch)
2. Why AI Engineer role?
3. Hardest technical problem you solved
4. Time you failed and what you learned
5. How you handle tight deadlines
6. Your biggest achievement at Digit7
7. Disagreement with teammate — how resolved
```

---

# Weekly Study Schedule

```
Monday:    ML/DL Theory (1 topic deep dive + notes)
Tuesday:   LLM/GenAI (1 topic + hands-on code)
Wednesday: DSA (1 problem) + Project mock interview
Thursday:  System Design (1 question, draw diagram)
Friday:    DSA (1 problem) + Weak area revision
Saturday:  Full mock interview (all topics, 1 hour)
Sunday:    Light revision only, rest
```

---

# Resources

```
ML/DL Theory:
  → StatQuest YouTube (best visual explanations)
  → fast.ai (practical deep learning)

LLM/GenAI:
  → LangChain docs (langchain.readthedocs.io)
  → DeepLearning.ai short courses (free, 1-2 hours each)
  → Anthropic Claude docs

DSA:
  → LeetCode (Easy → Medium only)
  → NeetCode YouTube (best explanations)

System Design:
  → "Designing ML Systems" — Chip Huyen (best ML system design book)
  → ML System Design Interview — YouTube

CV:
  → Papers with Code (latest SOTA)
  → Your own projects (biggest resource)
```

---

# Priority Order

```
WEEK 1-2 (High Priority):
  ✓ ML Fundamentals     → always asked, fills theory gap
  ✓ LLM/GenAI basics    → most important for AI Engineer
  ✓ DSA daily           → screening filter

WEEK 3-4 (High Priority):
  ✓ LLM/GenAI depth     → RAG, fine-tuning, evaluation
  ✓ System Design       → senior role differentiator
  ✓ DSA daily           → continue

WEEK 5 (Medium Priority):
  ✓ MLOps basics        → good to know
  ✓ Project mock interviews → polish answers
  ✓ Behavioral prep     → STAR stories

ONGOING:
  ✓ DSA 1/day           → never skip
  ✓ Project explanation → practice till fluent
```

---

# Your 60-Second Introduction

> "I'm Shanmugamani, a CV/ML Engineer with 4+ years of experience. At Digit7, I built production computer vision systems for an autonomous retail store — including a real-time product detection pipeline using SAM and ResNet that classifies products picked from shelves at 500ms latency across 30+ cameras. I also built a Pose Estimator that identifies which customer picked a product using YOLOX, Shapely polygon matching, and Xovis tracking.
>
> Recently I built a RAG-based AI chatbot using LangChain, Qdrant, and PostgreSQL for inventory queries and dispute management — with a 3-tier LLM fallback for reliability.
>
> I'm now looking to grow into an AI Engineer role — focusing on building production LLM systems, RAG pipelines, and scalable AI applications."

---

# Red Flags to Avoid in Interview

```
✗ "I don't know" without attempting → always try, show thinking
✗ Memorized answers without understanding → interviewer will dig deeper
✗ Only theory, no examples → always connect to your projects
✗ Saying POC is production → be honest, frame POC positively
✗ Not asking clarifying questions in system design → always clarify first
✗ Skipping edge cases → always mention failure scenarios
```
