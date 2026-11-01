# Day 7 — Feature Detection: HOG, SIFT, ORB
**Date:** 2026-08-05 | **Phase:** CV Revision | **Time:** 1 hour

---

## What is a Feature?

**Technical:**
A feature = distinctive point/region reliably found again even under rotation, scale, lighting change. Good features are at corners, blobs, distinctive textures.

**Analogy:**
> Two photos of same room from different angles. You match them by finding landmarks — clock, chair corner, painting. Those = features. Blank walls = useless (not distinctive).

---

## HOG — Histogram of Oriented Gradients

**Technical — Steps:**
```
1. Divide image into CELLS (8×8 pixels)
2. Each cell: compute gradient direction of every pixel
3. Build histogram of gradient directions (9 bins, 0°–180°)
4. Group 2×2 cells → BLOCK → normalize histogram (reduces lighting effect)
5. Concatenate all block histograms → HOG descriptor vector
```

For 64×128 image → **3780-dimensional descriptor vector**

```python
hog = cv2.HOGDescriptor()
descriptor = hog.compute(gray_img)   # returns HOG vector

# For pedestrian detection (original HOG use):
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(img, winStride=(8,8))
```

**Analogy:**
> Summarize a crowd by asking: "Which direction are most people facing?" Count per direction per block → that count = HOG descriptor. Same crowd layout = same histogram = match.

**Use cases:** Pedestrian detection, face detection, object classification (before deep learning).

---

## SIFT — Scale-Invariant Feature Transform

**Technical — 4 Steps:**

**Step 1: Scale-Space (DoG Pyramid)**
```
Blur image at increasing σ: σ=1, 2, 4, 8...
DoG = Gaussian(σ) − Gaussian(2σ)   → highlights blobs at each scale
```

**Step 2: Find Keypoints**
```
Local maxima in DoG across scale + space
Keypoint = max among 26 neighbours (8 same scale + 9 above + 9 below)
```

**Step 3: Assign Orientation**
```
Compute gradient directions in patch around keypoint
Dominant direction → keypoint orientation → rotation invariant
```

**Step 4: 128-dim Descriptor**
```
16×16 patch → 4×4 grid of cells → 8-bin histogram per cell
4 × 4 × 8 = 128 floats = SIFT descriptor
```

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
# keypoints: list of cv2.KeyPoint (x, y, size, angle, response)
# descriptors: (N, 128) float array
```

**Analogy:**
> Fingerprint at any zoom level.
> Scale-space = examine under different magnifications.
> Keypoint = distinctive whorl found at specific zoom.
> Orientation = rotate to standard position before comparing.
> 128-dim = the fingerprint's unique pattern code.

---

## ORB — Fast Alternative to SIFT

**Technical:**
```
FAST detector: circle of 16 pixels around candidate
               9+ consecutive brighter/darker → corner = keypoint

BRIEF descriptor: randomly compare 128 pixel pairs in patch
                  A brighter than B → 1, else → 0 → 128-bit binary string

Oriented BRIEF: rotate patch by keypoint direction first → rotation invariant
```

```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
# descriptors: (N, 32) uint8 — 32 bytes = 256 bits (binary)
```

**Analogy:**
> SIFT = detailed fingerprint (accurate, slow).
> ORB = quick face sketch (less detail, 100× faster).
> FAST = "Is this pixel dramatically different from its circle of neighbours? Yes → keypoint."
> BRIEF = 128 yes/no questions: "Is pixel A brighter than pixel B?" → binary code.

---

## HOG vs SIFT vs ORB

| Property | HOG | SIFT | ORB |
|----------|-----|------|-----|
| Type | Dense (whole image) | Sparse keypoints | Sparse keypoints |
| Speed | Medium | Slow | Very fast ✅ |
| Scale invariant | ❌ | ✅ | Partial |
| Rotation invariant | ❌ | ✅ | ✅ |
| Descriptor | 3780 floats | 128 floats | 256 bits |
| Best for | Detection (sliding window) | Matching, panorama | Real-time matching |

---

## Feature Matching

**Technical:**
```
BFMatcher: compare every descriptor against all others → exact nearest neighbour
FLANN:     approximate nearest neighbour → faster for large descriptor sets

Lowe's Ratio Test:
  best_match / second_best_match < 0.75 → good match
  → Filters ambiguous matches
```

```python
# BFMatcher for ORB (binary → Hamming distance)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# BFMatcher with ratio test for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2)
raw = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in raw if m.distance / n.distance < 0.75]
```

**Analogy:**
> 500 fingerprints from scene A vs 500 from scene B.
> BFMatcher = compare every A against ALL 500 B → accurate but slow.
> FLANN = smart index → narrows search fast.
> Ratio test = "Is my best match MUCH better than second best?" Yes → confident. No → reject.

---

## Quick Reference Cheatsheet (Day 7)

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Feature | Distinctive point reliably found again | Landmark in a room — clock, painting |
| HOG | Histogram of gradient directions per cell | Count which direction crowd faces per block |
| HOG cell | 8×8 pixel region | One section of the crowd |
| HOG block normalization | Normalize 2×2 cells together | Adjust for different lighting per area |
| SIFT scale-space | DoG pyramid at multiple blur levels | Fingerprint at different magnifications |
| SIFT keypoint | Local max across scale + space | Distinctive whorl in fingerprint |
| SIFT orientation | Dominant gradient direction | Rotate fingerprint to standard position |
| SIFT 128-dim | 4×4 cells × 8-bin histograms | Fingerprint unique pattern code |
| ORB FAST | Circle check — 9 consecutive pixels differ → corner | Spot who stands out in a circle of people |
| ORB BRIEF | 128 pixel pair comparisons → binary | 128 yes/no questions → binary ID |
| BFMatcher | Compare every pair | Compare every fingerprint vs every other |
| Ratio test | Best/second < 0.75 → good match | Best candidate much better than runner-up |

---

## Code File
[day07_feature_detection.py](../day07_feature_detection.py)

---

## Next: Day 8
**Topic:** CNN Internals — receptive field, backprop through conv, why CNNs work for images
