# Infosys Interview Preparation Plan
## Interview Date: Saturday, October 17th 2026

---

## Interview Structure (Experienced Candidates — 2-3 Rounds)

```
Round 1: Technical Interview
  → ML fundamentals
  → Deep Learning concepts
  → Computer Vision (your projects)
  → Python coding (1-2 problems)
  → RAG + LangChain + GenAI
  → Project walkthrough (MOST IMPORTANT)

Round 2: Behavioral / Managerial
  → Tell me about yourself
  → Challenges faced and how resolved
  → Team collaboration examples
  → Why Infosys?

Round 3: HR
  → Salary expectations
  → Notice period
  → Joining date
  → Relieving letter status
```

---

## 3-Day Preparation Plan

---

## Day 1 — October 14 (Tuesday) — Theory Revision

```
Morning: ML Fundamentals
  ✓ Supervised vs Unsupervised
  ✓ Overfitting vs Underfitting
  ✓ Loss functions (MSE, Cross Entropy, Focal)
  ✓ Optimizers (SGD, Adam, AdamW)
  ✓ Learning rate (too high vs too low)
  ✓ Batch Norm, Dropout
  ✓ Gradient, Vanishing gradient
  → Remaining: Precision, Recall, F1, Cross Validation

Afternoon: Deep Learning
  → CNN — how convolution works
  → Backpropagation
  → Transfer learning (fine-tuning vs feature extraction)
  → ResNet skip connections
  → EfficientNet, MobileNet
  → Attention mechanism

Evening:
  → DSA (1 problem)
  → Python basics:
    → List comprehension
    → OOP (class, inheritance, polymorphism)
    → Decorators, generators
```

---

## Day 2 — October 15 (Wednesday) — Projects + GenAI

```
Morning: Project Explanations (MOST IMPORTANT)
  → Product Handler — 2 min pitch
  → Pose Estimator — 2 min pitch
  → AI Chatbot — 2 min pitch
  → QC problem story (SAM + confidence threshold)
  → Practice each 3 times out loud

Afternoon: GenAI / RAG Revision
  → RAG pipeline (chunk → embed → store → retrieve → generate)
  → Chunking strategies
  → LangChain agent + tools + memory
  → Hallucination — causes and prevention
  → Token economics ($3/M input, $15/M output)
  → LLM fallback (Groq → Gemini → Ollama)
  → Prompt engineering (zero-shot, few-shot, CoT)

Evening:
  → DSA (1 problem)
  → CV concepts quick revision:
    → MOG2 background subtraction
    → SAM segmentation
    → YOLOX person detection
    → Homography / coordinate transformation
    → Fisheye undistortion
    → Edge vs cloud deployment
```

---

## Day 3 — October 16 (Friday) — Mock Interview + Rest

```
Morning: Full Mock Interview
  → Tell me about yourself (practice 5 times)
  → Project explanation full flow
  → 5 ML theory questions
  → 1 coding problem (DSA)

Afternoon: Behavioral Answers
  → Why Infosys?
  → Biggest technical challenge?
  → Team conflict — how resolved?
  → Biggest achievement at Digit7?
  → Where do you see yourself in 3 years?

Evening:
  → Light revision only
  → Keep documents ready:
    → Resume (Shanmugamani_Resume.pdf)
    → Certificates
    → Relieving letter status answer ready
  → Sleep early — REST IS IMPORTANT
```

---

## High Priority Topics (Definitely Asked)

```
✓ Tell me about yourself (2 min — memorize)
✓ Project explanation (Product Handler + Chatbot)
✓ Supervised vs Unsupervised
✓ Overfitting — how to detect and fix
✓ CNN — how it works
✓ Transfer learning
✓ What is RAG — full pipeline
✓ Python coding (easy/medium DSA)
✓ Precision, Recall, F1
✓ Batch norm, Dropout
```

---

## Medium Priority Topics

```
→ Gradient, Backpropagation
→ Vanishing gradient + ResNet fix
→ LangChain agent creation
→ Edge deployment vs cloud
→ Scaling (100 → 10k users)
→ YOLO, SAM, MOG2 concepts
```

---

## Your Strongest Points — Use Every Answer

```
1. Real production experience (rare for 4 years)
   "In my production system at Digit7..."

2. Real numbers (impresses interviewers):
   → 1.5 sec cloud vs 500ms edge
   → 95% accuracy with HITL fallback
   → 30+ cameras, real retail store
   → ₹24,000/month cost calculation

3. End-to-end ownership:
   "I built from scratch —
    data collection → training → deployment"

4. Both CV + GenAI experience:
   → Computer vision (Product Handler, Pose Estimator)
   → GenAI (RAG chatbot, LangChain, Qdrant)
   → Very rare combination → Infosys will value this
```

---

## 60-Second Introduction (Memorize This)

> "I'm Shanmugamani, CV/ML Engineer with 4+ years at Digit7.
>
> I built production computer vision systems for an autonomous retail store. The Product Handler detects which product was picked from the shelf using SAM segmentation and ResNet classifier — running at 500ms latency across 30+ cameras. I also built a Pose Estimator that identifies which customer picked the product using YOLOX detection and Xovis people tracking.
>
> Recently I built a RAG-based AI chatbot using LangChain, Qdrant, and PostgreSQL — handling inventory queries, dispute management, and product recommendations with a 3-tier LLM fallback for reliability.
>
> I'm looking to grow into an AI Engineer role, applying my production CV and GenAI experience to larger scale systems."

---

## Why Infosys? (Prepare This Answer)

> "Infosys is working on large-scale AI transformation projects across industries. My background in both computer vision and GenAI aligns well with the enterprise AI engineering problems Infosys solves for clients. I want to apply my hands-on production experience to larger scale, more diverse AI systems and grow with a global organization."

---

## Behavioral Answers — STAR Format

### Biggest Technical Challenge
> "In the Product Handler, the classifier was giving wrong predictions for visually similar products like Pepsi 500ml vs 1L — causing wrong billing. I analyzed the confusion matrix per class, identified failing products, improved the input using SAM segmentation instead of raw bounding boxes, and added a confidence threshold with HITL fallback. Result: accuracy crossed 95%."

### Biggest Achievement
> "Built a complete autonomous store CV system — Product Handler and Pose Estimator — that runs in production across 30+ cameras at 500ms latency. The system handles real customer billing with no staff involvement. Edge deployment reduced latency by 67% compared to cloud."

### Why Leaving Current Job
> "I've built strong production CV systems at Digit7 and recently expanded into GenAI/RAG. I want to apply both skills to larger scale problems and grow into a senior AI Engineer role with broader impact."

### Relieving Letter Status
> "I have formally initiated my resignation and sent the relieving mail to HR. I am awaiting their confirmation and will share the relieving letter before my joining date."

---

## Python Quick Revision

```python
# OOP
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof"

# List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Decorator
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start}")
        return result
    return wrapper

# Generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

---

## DSA Problems to Revise Before Saturday

```
✓ Two Sum           → HashMap
✓ Valid Parentheses → Stack
✓ Maximum Subarray  → Kadane's
✓ Contains Duplicate→ HashSet
✓ Move Zeroes       → Two Pointer
→ Find Pivot Index  → Prefix Sum (do this today)
```

---

## Day of Interview — October 17

```
Before interview:
  → Read this document once
  → Practice introduction 3 times
  → Keep resume ready
  → Be calm — you have REAL production experience

During interview:
  → Always connect theory to your projects
  → Mention real numbers (500ms, 95%, 30+ cameras)
  → If you don't know → say "I'll approach it this way..."
  → Ask clarifying questions in system design
  → Never say "I don't know" without attempting

After technical round:
  → Ask: "What does the AI team work on at Infosys?"
  → Shows genuine interest
```

---

## Topic Reference Files

```
Product Handler flow  → /nahs/product_handler_readme.md
Pose Estimator flow   → /nahs/pose_estimator_readme.md
AI Chatbot flow       → /nahs/project_chatbot_flow.md
CV Question Bank      → /nahs/cv_question_bank.md
ML Roadmap            → /nahs/interview_roadmap.md
DSA Solutions         → /nahs/dsa.ipynb
```
