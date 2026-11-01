# Product Handler — System Understanding

## What is this system?

Real-time computer vision system for autonomous retail stores.
When a customer picks up or puts back a product, the system:
1. Detects the action (via IoT sensor)
2. Captures camera frames before and after
3. Identifies what product was picked/put back
4. Stores result for automated billing

---

## Full System Flow

```
Customer picks up product
      ↓
IoT Sensor (weight sensor / Shekel) fires
      ↓
Redis (IoT DB) ← trigger stored { shelf_id, timestamp, action }
      ↓
Monitor Triggers (polls Redis every 100ms)
      ↓
Fetch Before + After frames from RTSP shelf camera
      ↓
Motion Detection (MOG2 Background Subtractor)
  → Scan all frames in buffer
  → Find last quiet frame before motion  = BEFORE image
  → Find first quiet frame after motion  = AFTER image
      ↓
SAM Model (Segment Anything Model)
  → Compare BEFORE vs AFTER
  → Find changed regions → bounding boxes
      ↓
Product Classifier (ResNet / MultiClass)
  → For each bounding box → predict product SKU
  → Returns: product name, confidence score, shelf index
      ↓
Redis (Product DB) ← store result
  { product_name, confidence, shelf_index, action: Pickup/Putback }
      ↓
Billing system reads from Redis → charges customer
```

---

## Key Components

### 1. Camera Management
```
- Each shelf has 1 camera (RTSP stream)
- PyAV library reads RTSP stream
- One thread per camera (parallel)
- Frame buffer: Queue (max 80 frames)
- Camera health monitor thread:
    checks all cameras every second
    if camera unresponsive > 2 seconds → auto-restart
```

### 2. Shelf Masking
Each camera covers multiple shelves. Mask defines which region in frame = which shelf.

```
Shelf Types:
  Gondola  → standard shelf, fixed mask coordinates
  Yilai    → custom mask from JSON file
  Shekel   → weight sensor shelf, camera confirms product
  Cooler   → refrigerated shelf (OpenCooler module)
```

### 3. Action Window Detection (MOG2)
```
Problem: We have 80 frames in buffer, which 2 are before/after?

Solution:
  → Apply MOG2 background subtractor to each frame
  → Compute motion score per frame (count of moving pixels)
  → Find last frame with low motion = BEFORE image
  → Find first frame with low motion after action = AFTER image
  → Fallback: use latest frame from live camera buffer
```

### 4. AI Pipeline
```
Step 1: SAM Model
  Input:  before image + after image + shelf mask
  Output: bounding boxes of changed regions
          (where product was picked/placed)

Step 2: Product Classifier
  Input:  cropped image of changed region (bounding box)
  Output: product SKU + confidence score

  Two classifier options:
    ResNet      → single product classification
    MultiClass  → multiple product classification
```

### 5. Redis DB Structure
```
REDIS_CONFIGURATION_DB → store config, camera IDs, product SKUs, shelf layout
REDIS_IOT_DB           → incoming triggers from IoT sensors
REDIS_PRODUCT_DB       → output detection results
REDIS_HEALTH_DB        → camera health, app health status
```

### 6. Multi-threading
```
Main thread         → startup, initialization
Camera threads      → one per camera (frame capture)
Health monitor      → one thread, watches all cameras
Trigger monitor 1   → DigitShelf (weight sensor triggers)
Trigger monitor 2   → Shekel triggers
Worker pool         → 10 threads, process triggers in parallel
```

---

## Memory Management (Production Optimization)
```python
MALLOC_ARENA_MAX = 2        # limit glibc memory arenas → reduce RAM fragmentation
gc.collect()                # Python garbage collection after each trigger
malloc_trim(0)              # force OS to reclaim unused memory
del before_image, after_image, mask  # explicit numpy array deletion
```

Why this matters: Edge device has limited RAM. Each trigger processes large images. Without cleanup, RAM fills up over hours of operation.

---

## Interview Explanation (60 seconds)

> "Product Handler is the core CV system for Digit7's autonomous stores. When a customer picks up a product, an IoT weight sensor triggers the pipeline. The system captures before and after frames from the shelf camera using PyAV RTSP streaming. We use MOG2 background subtraction to find the exact action window — the quiet frame before motion and after motion. SAM model then compares these two frames and segments the changed region. Finally, our ResNet classifier identifies the product SKU from that region with a confidence score. Results go to Redis and the billing system charges the customer automatically. The whole pipeline runs in under 3 seconds."

---

## Resume Bullets (Digit7)

```
● Product Handler — Autonomous Retail CV System:
  Built real-time product detection pipeline for autonomous stores —
  IoT sensor triggers → RTSP camera frame capture (PyAV) →
  MOG2 motion detection for action window → SAM segmentation →
  ResNet classification → Redis result storage for automated billing.
  End-to-end pipeline runs under 3 seconds per trigger.
  Tools: OpenCV, SAM, ResNet, PyTorch, PyAV, Redis, Docker

● Built multi-threaded camera management with auto-restart health
  monitor handling 10+ concurrent shelf cameras with zero manual
  intervention on camera failure.

● Improved product identification accuracy from 80% to 95% by
  optimizing shelf masking, action window detection, and
  classifier fine-tuning on store-specific product dataset.
```

---

## Key Terms to Know

| Term | Meaning |
|------|---------|
| RTSP | Real-Time Streaming Protocol — camera video stream |
| PyAV | Python library to read RTSP streams |
| MOG2 | Background subtraction algorithm for motion detection |
| SAM | Segment Anything Model — finds changed regions |
| ResNet | CNN classifier — identifies product from image |
| Queue | Frame buffer — stores last 80 frames per camera |
| Redis | In-memory DB — stores triggers, results, health data |
| Trigger | IoT sensor event when product is picked/placed |
| Shelf mask | Image defining which pixels = which shelf |
| Action window | Before image + after image pair for one customer action |
