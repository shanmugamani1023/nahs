# Day 1 — Image as Math: 2D Convolution
**Date:** 2026-07-31 | **Phase:** CV Revision | **Time:** 1 hour

---

## What is an Image?

> **Analogy:** An image is like a spreadsheet. Each cell is a pixel. The cell value is the brightness (0 = black, 255 = white). A color image is 3 spreadsheets stacked — one each for Red, Green, Blue.

- **Grayscale:** 2D NumPy array of shape `(H, W)` — values 0–255
- **Color:** 3D array of shape `(H, W, 3)` — BGR (OpenCV) or RGB (matplotlib)
- **Float image:** normalized to 0.0–1.0 for neural networks
- **dtype matters:** `uint8` for display, `float32/64` for computation

```python
img = cv2.imread("image.jpg")     # shape: (H, W, 3), dtype: uint8, BGR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
img_float = gray.astype(np.float32) / 255.0   # normalized float
```

---

## 2D Convolution Formula

> **Analogy:** Convolution is like using a stamp. You press the stamp at every position on the image, and it leaves a mark based on what pixels are underneath it. The stamp design (kernel) decides what feature gets highlighted.

```
Output[i, j] = Σ_m Σ_n  Image[i+m, j+n] × Kernel[m, n]
```

**Step by step:**
1. Place the kernel over a patch of the image
2. Element-wise multiply kernel × patch
3. Sum all values → one output pixel
4. Slide kernel to next position (stride=1), repeat

**Output size** (no padding): `(H - K + 1) × (W - K + 1)`  
**Output size** (SAME padding): `H × W` (pad = K // 2)

---

## True Convolution vs Cross-Correlation

> **Analogy:** True convolution is like reading a book in a mirror — the text is flipped. Cross-correlation is reading it normally. For symmetric books (palindromes), it makes no difference.

| | Formula | Kernel flip? |
|--|---------|-------------|
| True Convolution | `Image[i-m, j-n] × Kernel[m,n]` | Yes (180°) |
| Cross-Correlation | `Image[i+m, j+n] × Kernel[m,n]` | No |

**Reality:**
- `cv2.filter2D` → cross-correlation (no flip)
- `torch.nn.Conv2d` → cross-correlation (no flip, but learns kernel so flip doesn't matter)
- For **symmetric kernels** (Gaussian, Laplacian): identical result

---

## Kernels and Their Effects

> **Analogy:** A kernel is like a lens on a camera. A blur lens smears light from neighbours. An edge lens blocks flat light and only lets sharp changes through. The lens design (kernel values) determines what you see.

| Kernel | Values Sum | Effect |
|--------|-----------|--------|
| Identity | 1 | No change |
| Box Blur | 1 | Average of neighborhood |
| Gaussian Blur | 1 | Weighted average, less ringing than box |
| Edge Detect | 0 | High where contrast changes, 0 on flat areas |
| Sharpen | 1 | Center amplified, subtracts blurred version |
| Sobel X | 0 | Horizontal edges (vertical gradient) |
| Sobel Y | 0 | Vertical edges (horizontal gradient) |

**Key rule:**
- Kernel sum = 1 → brightness preserved
- Kernel sum = 0 → highlights change only (edges appear on dark background)

### Common Kernels

```python
# Gaussian (approximation)
[[1, 2, 1],
 [2, 4, 2],   ÷ 16    →  sum = 1.0
 [1, 2, 1]]

# Edge detection (Laplacian-like)
[[-1, -1, -1],
 [-1,  8, -1],          →  sum = 0.0
 [-1, -1, -1]]

# Sobel X (horizontal edges)
[[-1, 0, 1],
 [-2, 0, 2],            →  sum = 0.0
 [-1, 0, 1]]
```

---

## Padding

> **Analogy:** Padding is like adding a picture frame around a photo before cropping. Without it, the edges get cut off (output shrinks). SAME padding keeps the photo the same size. REFLECT padding mirrors the edge pixels — like placing the photo next to a mirror, so the border blends naturally.

```
VALID padding:   no padding → output shrinks by (K-1) pixels on each side
SAME padding:    pad = K//2 → output same size as input (zero-pad)
REFLECT padding: mirror border pixels → no edge artifacts (better than zero-pad)
```

---

## Receptive Field

> **Analogy:** Receptive field is like peripheral vision. One layer of conv = tunnel vision (only sees 3×3). Stack more layers = wider awareness. After 5 layers, one output neuron "sees" an 11×11 region — like slowly opening your eyes wider and wider.

| Layers (3×3 conv each) | Receptive Field |
|------------------------|----------------|
| 1 layer | 3 × 3 |
| 2 layers | 5 × 5 |
| 3 layers | 7 × 7 |
| n layers | (2n+1) × (2n+1) |

**Why this matters:**  
Two stacked 3×3 convs = same receptive field as one 5×5 conv  
But: 2×(3×3) = **18 params** vs 1×(5×5) = **25 params**  
→ Deeper networks with small kernels are more parameter-efficient

---

## Why Convolution Works for Vision

> **Analogy for Parameter Sharing:** Teaching a security guard to recognize a face once — and then that same recognition skill works at every door in the building. You don't train a separate guard for each door.

> **Analogy for Translation Equivariance:** If you move a cat from the left side of the photo to the right side, you still recognize it as a cat. The detector doesn't need to relearn for each position.

> **Analogy for Local Connectivity:** To tell if a pixel is an edge, you only need to look at its neighbours — not at pixels on the other side of the image.

| Property | Meaning | Why It Matters |
|----------|---------|----------------|
| Local connectivity | Output pixel only uses nearby pixels | Patterns (edges, textures) are spatially local |
| Parameter sharing | Same kernel applied everywhere | Cat-ear detector works at any image location |
| Translation equivariance | Shift input → shift output | Object detection doesn't need position-specific detectors |

**Why fully connected layers fail on images:**
> **Analogy:** FC layers treat an image like a bag of random numbers with no spatial meaning — like shuffling all the pixels randomly and trying to recognize a face. Conv layers preserve the spatial neighbourhood.

- A 224×224 image flattened = 150,528 inputs → too many parameters
- A cat in top-left vs bottom-right would need separate learned weights in FC

---

## Implementation from Scratch

```python
def convolve2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    # Reflect padding (mirrors border, no black edges)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    output = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)  # dot product
    return output
```

**Vectorized (fast) version using stride tricks:**
```python
from numpy.lib.stride_tricks import as_strided

patches = as_strided(padded, shape=(H, W, kH, kW), ...)
output = np.einsum('ijmn,mn->ij', patches, kernel)
```

---

## Key Takeaway

> Every `cv2.GaussianBlur()`, `cv2.Canny()`, `nn.Conv2d()` runs this formula.
> In classical CV, kernels are **hand-crafted**.
> In deep learning, kernels are **learned by backpropagation**.
> The math is the same.

---

## Quick Analogy Cheatsheet (Day 1)

| Concept | Analogy |
|---------|---------|
| Image | Spreadsheet — each cell is a pixel value |
| Convolution | Stamp sliding over every position of the image |
| Kernel | A lens — its design decides what feature you see |
| Blur kernel | Taking the average rating of nearby restaurants |
| Edge kernel | Spotting who is different in a group of similar people |
| Padding | Picture frame — keeps the photo the same size |
| Receptive field | Peripheral vision that widens as you stack more layers |
| Parameter sharing | One trained guard recognizes the same face at every door |
| Translation equivariance | Recognizing a cat on left or right — same skill, different place |
| Conv vs FC | Conv reads a map, FC shuffles the map into random numbers |

---

## Code File
[day01_convolution.py](../day01_convolution.py)

---

## Next: Day 2
**Topic:** Color & Histograms — RGB↔HSV math, histogram equalization from scratch
