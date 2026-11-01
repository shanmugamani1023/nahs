# Day 6 — Geometric Transforms: Affine, Perspective & Homography
**Date:** 2026-08-05 | **Phase:** CV Revision | **Time:** 1 hour

---

## What are Geometric Transforms?

**Technical:**
Change the **position/location** of pixels using a matrix — not their brightness. Every output pixel is mapped from an input pixel via math.

**Analogy:**
> Normal filters (blur, edge) change pixel **color**. Geometric transforms change pixel **location** — like moving, rotating or stretching a photo on a table.

---

## Types of Transforms

| Transform | Matrix | Points Needed | Preserves |
|-----------|--------|--------------|-----------|
| Translation | 2×3 | — | Shape, size, angles, parallel lines |
| Rotation | 2×3 | — | Shape, size, angles, parallel lines |
| Affine | 2×3 | 3 pairs | Parallel lines, distance ratios |
| Perspective | 3×3 | 4 pairs | Straight lines only |
| Homography | 3×3 | 4+ pairs (auto) | Straight lines only |

---

## Affine Transform

**Technical:**
2×3 matrix — handles translation, rotation, scale, shear in one matrix.
```
[x']   [a  b  tx] [x]
[y'] = [c  d  ty] [y]
                  [1]
```
- Parallel lines → always remain parallel
- 3 point pairs needed to compute matrix

```python
src = np.float32([[0,0], [100,0], [0,100]])
dst = np.float32([[10,5], [110,15], [5,110]])

M      = cv2.getAffineTransform(src, dst)
result = cv2.warpAffine(img, M, (W, H))

# Rotation specifically:
M = cv2.getRotationMatrix2D(center=(W//2, H//2), angle=45, scale=1.0)
result = cv2.warpAffine(img, M, (W, H))
```

**Analogy:**
> Affine = rubber sheet with image printed on it. Stretch, rotate, skew freely — but railway tracks always remain parallel tracks. They may change angle but never converge.

---

## Perspective Transform

**Technical:**
3×3 matrix — handles full camera angle correction.
```
[x']     [h11  h12  h13] [x]
[y']  =  [h21  h22  h23] [y]    → divide by w' to get actual (x', y')
[w']     [h31  h32  h33] [1]
```
- Straight lines stay straight
- Parallel lines CAN converge (vanishing point)
- 4 point pairs needed

```python
# Document scanner example:
src = np.float32([[50,50], [300,30], [320,450], [20,470]])  # skewed corners
dst = np.float32([[0,0],   [400,0],  [400,600], [0,600]])   # flat rectangle

M      = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, M, (400, 600))
```

**Analogy:**
> Perspective = looking at a rectangular table from an angle — it appears as a trapezoid. Perspective transform corrects it back to a rectangle, as if you're looking from directly above. This is exactly what a document scanner app does.

---

## Affine vs Perspective

**Technical:**
```
Affine:      parallel → parallel always
Perspective: parallel → may converge at vanishing point

Affine needs 3 points  (6 equations for 6 unknowns in 2×3 matrix)
Perspective needs 4 points (8 equations for 8 unknowns in 3×3 matrix)
```

**Analogy:**
> **Affine** = pulling rubber sheet corners — railway tracks stay parallel always.
> **Perspective** = viewing rubber sheet from an angle — tracks appear to meet at horizon (vanishing point).

---

## Homography

**Technical:**
Same 3×3 matrix as perspective — but computed **automatically** from matched feature points using RANSAC (not manually picked).

```python
# From feature matching (ORB/SIFT):
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
result  = cv2.warpPerspective(img1, H, (W, H))
```

RANSAC = automatically rejects wrong/outlier matches.

**Analogy:**
> **Perspective** = you manually point to 4 corners of a document.
> **Homography** = algorithm automatically finds matching dots between two photos and computes the same correction on its own. Like having an assistant who finds the corners for you.

---

## Real Use Cases

| Situation | Transform | Why |
|-----------|-----------|-----|
| Document scanner | Perspective | Fix camera angle → flat rectangle |
| Bird's eye view (parking/lanes) | Perspective | Top-down view from angled camera |
| Data augmentation for training | Affine | Random rotate, scale, shear → more variety |
| Panorama stitching | Homography | Align two camera views automatically |
| AR overlay on surface | Homography | Map image onto detected plane |
| Template matching with angle | Affine | Correct slight tilt before matching |
| Inspection — align template | Homography | Auto-align reference vs captured image |

---

## Quick Reference Cheatsheet (Day 6)

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Geometric transform | Changes pixel location via matrix | Moving/rotating photo on a table |
| Affine (2×3) | Parallel lines preserved, 3 points needed | Rubber sheet — tracks stay parallel |
| Perspective (3×3) | Straight lines only, 4 points needed | Table seen from angle → trapezoid corrected |
| Homography | Auto perspective from feature matches | Assistant finds corners automatically |
| warpAffine | Apply 2×3 matrix to image | Execute the rubber sheet stretch |
| warpPerspective | Apply 3×3 matrix to image | Execute the angle correction |
| RANSAC in findHomography | Rejects outlier matches automatically | Ignores wrong feature matches |
| Vanishing point | Where parallel lines appear to meet | Train tracks meeting at horizon |

---

## Code File
[day06_geometric_transforms.py](../day06_geometric_transforms.py)

---

## Next: Day 7
**Topic:** Feature Detection — HOG math, SIFT scale-space, ORB, feature matching
