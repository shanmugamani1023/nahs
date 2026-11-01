# Day 8 — CNN Internals

## Concepts Covered
- Local connectivity
- Parameter sharing
- Receptive field
- Backprop through conv
- ResNet skip connections

---

## 1. Why Not Fully Connected?

224×224 image = 150,000 inputs × millions of weights → too slow, too much memory.

CNNs solve this with 3 ideas.

---

## 2. Local Connectivity

Each neuron sees only a small patch (e.g., 3×3), not the whole image.

**Technical:** 3×3 kernel connects to 9 pixels, not all 150,000.

**Analogy:** Security guard watches one zone, not the entire city.

---

## 3. Parameter Sharing

Same kernel slides across entire image.

**Technical:**
```
3×3 kernel = 9 weights (used at every position)
Instead of 224×224×9 = 451,584 weights → just 9
```

**Analogy:** One stamp used everywhere on the page. Same edge-detector works at top-left AND bottom-right.

---

## 4. Receptive Field

How much of the original image does one output pixel "see"?

**Formula:**
```
RF = 1 + (kernel_size - 1) × num_layers

Layer 1 (3×3): RF = 3
Layer 2 (3×3): RF = 5
Layer 3 (3×3): RF = 7
Layer 4 (3×3): RF = 9
```

**Analogy:** Reading through a 3-letter window. Pass 1 = see 3 letters. Pass 2 = effectively see 5. Deeper = wider view.

**Why it matters:** Deep layers detect global patterns (faces), shallow layers detect local (edges). RF grows automatically with depth.

---

## 5. Kernel = Weights

```
Kernel values = weights learned during training

Before training: random values (0.3, -0.1, 0.7...)
After training:  meaningful values (Sobel, Gabor, blob detectors...)
```

CNN doesn't use hand-crafted kernels — it learns them from data.

**Learning loop:**
```
Random kernel → apply to image → loss → backprop → update kernel → repeat
```

---

## 6. Backprop Through Conv

**Forward:**
```
output = sum(input_patch × kernel)
```

**Backward:**
```
∂Loss/∂kernel = convolve(input, upstream_gradient)
∂Loss/∂input  = convolve(upstream_gradient, flipped_kernel)
```

**Numerical example:**
```
Input patch:    1 2 3       Kernel: 1  0 -1
                4 5 6               2  0 -2
                7 8 9               1  0 -1

output = (1×1)+(2×0)+(3×-1) + (4×2)+(5×0)+(6×-2) + (7×1)+(8×0)+(9×-1) = -8

If target = -5, error = -3 (upstream gradient)

∂Loss/∂kernel[i,j] = input[i,j] × (-3)
kernel[0,0] gradient = 1 × (-3) = -3
kernel[0,2] gradient = 3 × (-3) = -9

weight update: kernel[i,j] = kernel[i,j] - lr × gradient
```

**Analogy:** Input pixel value = how loudly that ingredient "spoke." Larger input = that kernel weight gets adjusted more.

**Parameter sharing effect:** Kernel slides to 100 positions → gradients from ALL positions summed → one update. Like a teacher adjusting rubric after grading all 100 students.

---

## 7. Vanishing Gradient Problem

In deep networks (50+ layers), gradients get multiplied many times backward:
```
small × small × small × ... = ~0
```
Early layers learn nothing.

**Analogy:** Whisper through 50 people → message completely garbled by person 1.

---

## 8. ResNet Skip Connections

```
Normal:  x → Conv → ReLU → Conv → ReLU → output
ResNet:  x → Conv → ReLU → Conv → ReLU → (+x) → output
```

`output = F(x) + x`

**Gradient through skip:**
```
∂(F(x) + x)/∂x = ∂F(x)/∂x + 1
```

Even if `∂F(x)/∂x → 0`, gradient is still **1** → always flows.

**Analogy:** Direct phone call to person 1 alongside the whisper chain. Original message always arrives intact.

---

## Quick Reference

| Concept | Technical | Analogy |
|---------|-----------|---------|
| Local connectivity | 3×3 patch only | Guard watches one zone |
| Parameter sharing | Same kernel everywhere | One stamp, all pages |
| Receptive field | RF = 1 + (k-1) × layers | Window widens each pass |
| Kernel = weights | 9 numbers learned by backprop | Stamp shape carved by training |
| Vanishing gradient | Small × small × ... → 0 | Whisper through 50 people |
| Skip connection | output = F(x) + x, grad has +1 | Phone call bypasses chain |

---

## Receptive Field Calculator

```python
def receptive_field(num_layers, kernel_size):
    return 1 + (kernel_size - 1) * num_layers

print(receptive_field(5, 3))   # → 11
print(receptive_field(13, 3))  # → 27 (VGG-like)
```
