# Day 4 — Edge Detection: Gradients & Canny Pipeline
**Date:** 2026-08-03 | **Phase:** CV Revision | **Time:** 1 hour

---

## What is an Edge?

**Technical:**
An edge is a pixel location where brightness changes sharply — high rate of change between neighbouring pixels.
```
Flat area:  100  100  100  100  → difference = 0 → no edge
Edge:       100  100  255  255  → difference = 155 → EDGE HERE
```

**Analogy:**
> You are walking on a footpath. Exact point where smooth concrete becomes rough gravel = edge. Before that point → same surface (flat region). After → completely different.

---

## Gradient

**Technical:**
Gradient = rate of change of pixel intensity. Computed in X and Y directions separately.
```
Gx = Image[i, j+1] - Image[i, j-1]   → horizontal change → detects vertical edges  |
Gy = Image[i+1, j] - Image[i-1, j]   → vertical change   → detects horizontal edges ─

Magnitude  G = √(Gx² + Gy²)     → how strong the edge is
Direction  θ = arctan(Gy / Gx)   → which way the edge runs
```

**Analogy:**
> You are cycling on a road. Flat road = gradient 0 (no effort). Uphill = high gradient (more effort). The steeper the hill, the higher the gradient. Gradient magnitude = how steep. Gradient direction = which way the slope faces.

---

## Sobel Operator

**Technical:**
Sobel uses weighted differences. Center row/column gets 2× weight for smoothing effect.
```
Sobel X (detects vertical edges |):    Sobel Y (detects horizontal edges ─):
-1   0  +1                             -1  -2  -1
-2   0  +2    ← center row 2× weight   0   0   0
-1   0  +1                             +1  +2  +1

At a vertical edge (dark left, bright right):
Gx = (-1×0 + 0×? + 1×255) × weights = large positive value
At flat region:
Gx = (-1×100 + 0×? + 1×100) × weights = 0
```

**Analogy:**
> **Sobel X:** Two people stand on your left (−) and right (+). Left person pushes you back, right person pulls you forward. If both push equally → you stay (no edge). If right pulls much harder (bright right, dark left) → you move strongly → **vertical edge detected.**
>
> **Sobel Y:** Same idea, top person pushes up (−), bottom pulls down (+).

```python
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
G  = np.sqrt(Gx**2 + Gy**2)
```

---

## Canny Pipeline — 5 Steps

**Technical overview:**
5-step pipeline that produces thin, accurate, noise-free edges.

**Analogy:**
> Canny = airport security system. Clean the scanner → detect all items → pinpoint exact location → keep only real threats → verify weak signals.

---

### Step 1: Gaussian Blur

**Technical:**
Smooth image to remove high-frequency noise. Without this, noise pixels produce false large gradients → fake edges.
```python
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
```

**Analogy:**
> Before scanning luggage, wipe dust off the scanner. Dust marks (noise) would trigger false alarms.

---

### Step 2 + 3: Sobel Gradients → Magnitude + Direction

**Technical:**
Apply Sobel X and Y kernels → compute magnitude (edge strength) and direction (edge angle) at every pixel. At this stage edges are **thick** — multiple pixels fire at each edge boundary.
```python
G     = np.sqrt(Gx**2 + Gy**2)
theta = np.arctan2(Gy, Gx)
```

**Analogy:**
> Scanner detects a suspicious zone — but it highlights a whole region, not a precise object.

---

### Step 4: Non-Maximum Suppression (NMS)

**Technical:**
Thin thick edges to exactly 1 pixel width.
- For each pixel: look at 2 neighbours **along the gradient direction**
- If this pixel is NOT the local maximum → set to 0 (suppress)
- If it IS the local maximum → keep it

```
Before NMS (thick edge, 3 pixels wide):   After NMS (1 pixel thin):
  0    0    0                               0    0    0
 80  [120]  90    →                         0  [120]   0   ← only peak survives
  0    0    0                               0    0    0
```

**Analogy:**
> Three people all raise their hand saying "I am the edge!" NMS says — only the tallest person keeps their hand up. The other two sit down. Result = exactly 1 person standing = 1 pixel wide edge.

---

### Step 5: Double Threshold + Hysteresis

**Technical:**
Classify remaining pixels into 3 groups using two thresholds:
```
pixel > T_high          → STRONG edge → definitely keep ✅
pixel < T_low           → noise       → definitely discard ❌
T_low ≤ pixel ≤ T_high  → WEAK edge  → keep ONLY IF 8-connected to a strong edge pixel
```
Hysteresis = the step that checks connectivity between weak and strong edges.

**Analogy:**
> **Airport gate analogy:**
> - Passenger with full boarding pass + ID (> T_high) → definitely boards ✅ **(strong edge)**
> - Passenger with no documents (< T_low) → definitely rejected ❌ **(noise)**
> - Passenger with boarding pass but no ID (between) → boards ONLY IF standing next to a verified passenger **(weak edge + hysteresis)**

```
Strong edges:   ████████████
Weak connected: ░░░████░░░░░   → kept (touching strong)
Weak alone:     ░░░░░░░░████   → discarded (isolated)
```

---

## Why 2 Thresholds and Not 1?

**Technical:**

| Approach | Problem |
|----------|---------|
| Only high threshold | Misses real faint/thin edges |
| Only low threshold | Keeps too much noise |
| 2 thresholds + hysteresis | Strong edges kept, weak kept only if connected ✅ |

**Analogy:**
> One strict rule → miss real guests. One loose rule → let in gatecrashers. Two rules → best of both: confirm real guests, reject gatecrashers, give borderline cases a chance.

---

## Full Code

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

# Understand gradients manually:
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
G  = np.sqrt(Gx**2 + Gy**2)

# OpenCV handles NMS + hysteresis internally:
edges = cv2.Canny(blur, threshold1=50, threshold2=150)
# Rule of thumb: T_high = 2× or 3× T_low
```

---

## Quick Reference Cheatsheet (Day 4)

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Edge | Sharp brightness change between pixels | Footpath — concrete to gravel transition |
| Gradient | Rate of brightness change (Gx, Gy) | Slope of a cycling road |
| Sobel X | Weighted left−right pixel difference | Left person pushes back, right pulls forward |
| Sobel Y | Weighted top−bottom pixel difference | Top person pushes up, bottom pulls down |
| Gradient magnitude | √(Gx²+Gy²) = edge strength | Total pulling force from all directions |
| Gradient direction | arctan(Gy/Gx) = edge angle | Which way the slope faces |
| NMS | Keep only local maximum along gradient | Only tallest person in a row stays standing |
| High threshold | pixel > T_high = strong edge | Full boarding pass + ID → boards immediately |
| Low threshold | pixel < T_low = noise | No documents → rejected immediately |
| Weak edge | T_low ≤ pixel ≤ T_high | Boarding pass, no ID → needs verification |
| Hysteresis | Connected to strong edge → keep | Standing next to verified passenger → allowed |

---

## Code File
[day04_edge_detection.py](../day04_edge_detection.py)

---

## Next: Day 5
**Topic:** Morphological Operations — erosion, dilation, opening, closing
