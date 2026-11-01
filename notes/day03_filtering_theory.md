# Day 3 — Filtering Theory: Frequency Domain & Gaussian Derivation
**Date:** 2026-08-03 | **Phase:** CV Revision | **Time:** 1 hour

---

## What is a Filter?

> **Analogy:** A filter is like an audio equalizer. You can boost bass (low freq) or treble (high freq). Image filters do the same — but on pixel intensity waves instead of sound waves.

| Filter Type | Lets Through | Blocks | Effect |
|-------------|-------------|--------|--------|
| Low-pass | Low frequencies | High frequencies | Blur — removes detail |
| High-pass | High frequencies | Low frequencies | Edge detection — removes smooth areas |
| Band-pass | Middle range | Extremes | Selective enhancement |

---

## Frequencies in an Image

> **Analogy:** Scanning across a row of pixels — slow brightness change (sky) = low frequency like a bass note. Sharp brightness change (edge) = high frequency like a treble note.

```
Low frequency  → slow pixel changes  → smooth gradients, large regions, background
High frequency → fast pixel changes  → edges, fine textures, noise
```

| Operation | Frequency Effect |
|-----------|-----------------|
| Blur | Removes high freq → edges soften |
| Edge detection | Keeps only high freq → smooth areas → 0 |
| Noise | Unwanted high freq signal |

---

## Fourier Transform (Intuition)

> **Analogy:** A song = mix of many pure notes. Fourier Transform breaks the song into individual notes and shows how loud each is. For images: breaks it into spatial frequency components.

```
Image (spatial domain)  ←FFT→  Frequency map (frequency domain)
```

In frequency map:
- **Center** = low frequencies (overall brightness, slow gradients)
- **Edges of map** = high frequencies (sharp details, noise, edges)

**Why this matters for filters:**
- Gaussian blur (spatial) = multiply by Gaussian (frequency) → smooth, clean cut
- Box blur (spatial) = multiply by sinc function → oscillations → **ringing artifacts**

---

## Gaussian Kernel — The Math

> **Analogy:** The Gaussian is a bell curve. Drop a pebble in still water — ripples spread and fade smoothly. A box kernel is like dropping a brick — sharp boundary causes unnatural ringing.

**2D Gaussian formula:**
```
G(x, y) = (1 / 2πσ²) × exp(−(x² + y²) / 2σ²)
```

- `σ` = spread (larger = more blur)
- Decays smoothly from center → no ringing
- Integrates to 1.0 → brightness preserved

**Discrete 3×3 approximation (σ ≈ 1):**
```
     1  2  1
1/16×2  4  2      center weight (4) highest, corners (1) least
     1  2  1
```

**Common misconception:** "Gaussian blurs only outer pixels."
**Reality:** It blurs ALL pixels. The weights tell how much each neighbour contributes to the CENTER pixel's output. Center has highest weight (4/16 = 25%) — its own value matters most. Corners have lowest weight (1/16 = 6.25%) — they contribute least.

```
Output[i,j] = 6.25%×TL + 12.5%×T + 6.25%×TR
            + 12.5%×L  + 25%×C   + 12.5%×R     ← C (center) dominates
            + 6.25%×BL + 12.5%×B + 6.25%×BR
```

> **Analogy:** You survey 9 people. You trust yourself (center) most (25%), direct neighbours somewhat (12.5%), and diagonal corners least (6.25%). You still get a say — just blended with others. That blending across every pixel = smooth blur.

**Separable property (key optimization):**
```
G(x,y) = G(x) × G(y)
```
So: apply 1D Gaussian on rows, then on columns.
- Cost: O(2N) per pixel instead of O(N²) → much faster for large kernels

---

## Sigma (σ) — Key Parameter

> **Analogy:** σ is the blast radius of a soft explosion. Small σ = tiny explosion, only immediate neighbours. Large σ = wide effect, far pixels still feel it.

| σ value | Effect |
|---------|--------|
| 0.5–1 | Gentle blur, preserves detail |
| 2–3 | Moderate blur, reduces noise |
| 5+ | Heavy blur, loses significant detail |

**OpenCV rule:** `ksize` should be odd and `≥ 6σ + 1`
```python
cv2.GaussianBlur(img, (15, 15), sigmaX=2.0)  # ksize=15 covers ~3σ each side
```

---

## Filter Comparison

> **Box blur analogy:** Average everyone's salary in a neighborhood — including the millionaire. He drags the mean up (bright noise pixel survives).

> **Gaussian blur analogy:** Weighted survey — nearby neighbors count more, far ones less. The outlier at the edge has reduced influence, but still some.

> **Median filter analogy:** Line everyone up by salary, pick the middle person. The millionaire gets pushed to one end — can't affect the median. Salt-and-pepper noise pixel = the outlier. Median kills it completely.

| Filter | Mechanism | Best For | Weakness |
|--------|-----------|---------|---------|
| Box blur | Simple average | Fast preview | Ringing, edges affected |
| Gaussian blur | Weighted average (Gaussian weights) | General smoothing | Still blurs edges |
| Median filter | Replace with median of neighbors | Salt & pepper noise | Slow, blurs fine textures |
| Bilateral | Gaussian in space AND in intensity | Noise + edge preservation | Very slow |

---

## Code: Which Filter for Which Noise?

```python
# Salt & pepper noise → use Median
denoised = cv2.medianBlur(noisy_img, ksize=3)

# Gaussian noise → use Gaussian blur
denoised = cv2.GaussianBlur(noisy_img, (5, 5), sigmaX=1.0)

# Preserve edges while denoising → use Bilateral
denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Fast rough blur → use Box
blurred = cv2.blur(img, (5, 5))
```

---

## Gaussian from Scratch

```python
def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()   # normalize so sum = 1.0

def gaussian_blur(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)   # from Day 1
```

---

## Quick Analogy Cheatsheet (Day 3)

| Concept | Analogy |
|---------|---------|
| Image filter | Audio equalizer — boost or cut frequencies |
| Low frequency | Bass — slow brightness changes (sky, wall) |
| High frequency | Treble — sharp brightness changes (edges, noise) |
| Fourier Transform | Break a song into individual notes |
| Gaussian shape | Pebble in water — smooth ripples fading outward |
| σ (sigma) | Blast radius — how far the blur reaches |
| Box blur | Average salary including the neighborhood millionaire |
| Gaussian blur | Weighted survey — nearby neighbors count more |
| Median filter | Pick the middle person — outliers can't influence |
| Bilateral filter | Blur only with pixels of similar brightness (respects edges) |

---

## Code File
[day03_filtering_theory.py](../day03_filtering_theory.py)

---

## Next: Day 4
**Topic:** Edge Detection — gradient math, Canny pipeline internals (NMS + hysteresis)
