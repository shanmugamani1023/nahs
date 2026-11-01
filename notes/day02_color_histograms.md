# Day 2 — Color Spaces & Histograms
**Date:** 2026-08-02 | **Phase:** CV Revision | **Time:** 1 hour

---

## Color Spaces

> **Analogy:** RGB is like describing a paint color by how much red, green, and blue paint you mixed. HSV is like describing it the way a painter thinks — "what hue is it, how vivid, how bright?" Same color, different language. HSV is more natural for masking because "orange" is one hue range, not a mix of R+G values.

| Space | Channels | Range (OpenCV) | Best For |
|-------|----------|---------------|---------|
| BGR | Blue, Green, Red | 0–255 each | Default in OpenCV |
| RGB | Red, Green, Blue | 0–255 each | matplotlib, PyTorch |
| HSV | Hue, Saturation, Value | H:0–179, S:0–255, V:0–255 | Color masking |
| Grayscale | Intensity | 0–255 | Edge detection, classical CV |

---

## RGB → Grayscale

> **Analogy:** Human eyes are not equally sensitive to all colors. We see green most (3× more green photoreceptors), red a bit, blue barely. Grayscale is a weighted average that matches human perception — not a simple average.

```
Gray = 0.299×R + 0.587×G + 0.114×B
```

Why not `(R+G+B)/3`?
- Simple average makes blue look too bright, green too dim
- Weights above match human vision (ITU-R BT.601 standard)

```python
# OpenCV does this internally:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Manual:
gray_manual = 0.299*R + 0.587*G + 0.114*B
```

---

## RGB → HSV Math

> **Analogy:** HSV is like a color wheel. Hue = angle on the wheel (what color). Saturation = how far from center (vivid vs washed out). Value = how bright the whole wheel is lit. "Red, vivid, bright" = H≈0, S=255, V=255.

**Step 1:** Normalize to 0–1
```
R' = R/255,  G' = G/255,  B' = B/255
```

**Step 2:** Find max, min, delta
```
Cmax = max(R', G', B')
Cmin = min(R', G', B')
Δ = Cmax - Cmin
```

**Step 3:** Hue (0–360°)
```
Δ = 0         → H = 0
Cmax = R'     → H = 60 × ((G'-B')/Δ mod 6)
Cmax = G'     → H = 60 × ((B'-R')/Δ + 2)
Cmax = B'     → H = 60 × ((R'-G')/Δ + 4)
```

**Step 4:** Saturation
```
Cmax = 0  →  S = 0
else      →  S = Δ / Cmax
```

**Step 5:** Value
```
V = Cmax
```

**OpenCV quirk:** H stored as 0–179 (not 0–360) to fit uint8.
So when masking red: use `H < 10` in OpenCV = `H < 20°` on real wheel.

### Why HSV is better for color masking

```python
# RGB: "orange" spans many R,G,B combos — hard to define a range
# HSV: "orange" = H in range [10, 25] — simple!

lower = np.array([10, 100, 100])   # H, S, V lower bound
upper = np.array([25, 255, 255])   # H, S, V upper bound
mask = cv2.inRange(hsv_image, lower, upper)
```

---

## Histogram

> **Analogy:** A histogram is a crowd count at a concert. X-axis = seat row number (brightness 0–255). Y-axis = how many people (pixels) sit in that row. Dark image = crowd packed in rows 0–50. Bright image = crowd in rows 200–255. Perfect exposure = crowd spread across all rows.

```
histogram[k] = number of pixels with intensity = k,  for k in 0..255
```

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])  # shape: (256,1)

# Manual:
hist = np.zeros(256)
for pixel_val in image.flatten():
    hist[pixel_val] += 1
# or vectorized:
hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
```

---

## Histogram Equalization

> **Analogy:** Spread the concert crowd evenly across all rows. Rows that are packed get stretched out. Empty rows get filled. The result: every brightness level is equally represented → maximum contrast.

**Step 1:** Compute histogram `h[k]`

**Step 2:** Compute CDF (cumulative distribution)
```
cdf[k] = h[0] + h[1] + ... + h[k]
```
> CDF = running total of pixels from brightness 0 up to k

**Step 3:** Normalize CDF to 0–255
```
cdf_norm[k] = round( (cdf[k] - cdf_min) / (N - cdf_min) × 255 )
```
where N = total number of pixels, cdf_min = first non-zero CDF value

**Step 4:** Remap every pixel
```
output[i,j] = cdf_norm[ image[i,j] ]
```

```python
# OpenCV:
eq = cv2.equalizeHist(gray)

# Manual implementation:
hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_min = cdf[cdf > 0].min()
N = gray.size
lut = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)
equalized = lut[gray]
```

---

## CLAHE

> **Analogy:** Regular equalization gives the whole stadium one global brightness setting — bad if section A is dark and section B is already bright. CLAHE gives each section its own dimmer switch. Each tile equalizes locally, but clips extremes to avoid amplifying noise.

- Splits image into tiles (e.g., 8×8 grid)
- Equalizes each tile independently
- **Clip limit:** caps histogram before equalizing (prevents over-amplifying noise)
- Bilinear interpolation between tiles removes hard borders

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
result = clahe.apply(gray)
```

**When to use which:**
| Method | Use When |
|--------|---------|
| `equalizeHist` | Low contrast globally (dark or washed-out image) |
| `CLAHE` | Uneven lighting (some areas dark, some bright) — almost always better |

---

## Quick Analogy Cheatsheet (Day 2)

| Concept | Analogy |
|---------|---------|
| RGB | Mixing paint — how much Red, Green, Blue |
| HSV | Painter's language — Hue (color), Saturation (vivid/pale), Value (bright/dark) |
| H stored 0–179 in OpenCV | OpenCV halves hue to fit in a uint8 byte |
| Grayscale weights (0.299/0.587/0.114) | Eyes see green most, red some, blue barely |
| Histogram | Crowd count per brightness row at a concert |
| CDF | Running total of crowd from row 0 to row k |
| Histogram Equalization | Spread crowd evenly across all seat rows |
| CLAHE | Each section of the stadium has its own dimmer switch |

---

## Code File
[day02_color_histograms.py](../day02_color_histograms.py)

---

## Next: Day 3
**Topic:** Filtering Theory — Gaussian kernel derivation, frequency domain intuition, Fourier basics
