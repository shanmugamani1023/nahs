"""
Day 3: Filtering Theory — Gaussian from scratch, compare filters on noisy images
Goal: Understand WHY Gaussian beats Box, and WHEN to use Median vs Gaussian
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: Gaussian kernel from scratch
# ─────────────────────────────────────────────

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    Formula: G(x,y) = exp(-(x²+y²) / 2σ²)  then normalize to sum=1
    """
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]          # coordinate grid centered at 0
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()                      # normalize → sum = 1.0 (preserves brightness)
    return kernel


def box_kernel(size: int) -> np.ndarray:
    """All weights equal — simple average."""
    return np.ones((size, size), dtype=np.float64) / (size * size)


# ─────────────────────────────────────────────
# STEP 2: Apply convolution (reuse Day 1 logic)
# ─────────────────────────────────────────────

def apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import as_strided
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    shape = (H, W, kH, kW)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    patches = as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijmn,mn->ij', patches, kernel)


# ─────────────────────────────────────────────
# STEP 3: Add different types of noise
# ─────────────────────────────────────────────

def add_gaussian_noise(image: np.ndarray, sigma: float = 25) -> np.ndarray:
    """Gaussian noise — random continuous values. Common in real cameras."""
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """Salt & pepper — random pixels set to 0 or 255. Common in transmission errors."""
    noisy = image.copy()
    mask_salt   = np.random.random(image.shape) < prob / 2
    mask_pepper = np.random.random(image.shape) < prob / 2
    noisy[mask_salt]   = 255
    noisy[mask_pepper] = 0
    return noisy


# ─────────────────────────────────────────────
# STEP 4: Sigma effect demonstration
# ─────────────────────────────────────────────

def demo_sigma_effect(gray: np.ndarray):
    print("\n── Sigma (σ) Effect on Gaussian Blur ──")
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        kernel = gaussian_kernel(size=int(6 * sigma + 1) | 1, sigma=sigma)
        blurred = apply_filter(gray.astype(float), kernel)
        sharpness = cv2.Laplacian(blurred.astype(np.uint8), cv2.CV_64F).var()
        print(f"  σ={sigma:.1f} → kernel size={kernel.shape[0]}×{kernel.shape[0]} | sharpness={sharpness:.1f}")
    print("  (Lower sharpness = more blur)")


# ─────────────────────────────────────────────
# STEP 5: Compare filters on different noise types
# ─────────────────────────────────────────────

def psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio — higher = better quality recovery."""
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    # Load or create test image
    img = cv2.imread("test.jpg")
    if img is None:
        print("No image found — creating synthetic test image")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 40:260] = [180, 180, 180]    # light gray rect
        img[70:130, 100:200] = [80, 80, 80]       # dark gray inner rect
        for i in range(200):
            img[i, :, :] = np.clip(img[i, :, :].astype(int) + i // 4, 0, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add noise
    gauss_noisy = add_gaussian_noise(gray, sigma=25)
    sp_noisy    = add_salt_pepper_noise(gray, prob=0.05)

    # ── Gaussian kernel validation ───────────────────────────────────
    k = gaussian_kernel(size=5, sigma=1.0)
    print(f"Gaussian kernel sum: {k.sum():.6f}  (should be 1.0)")
    print(f"Gaussian kernel center weight: {k[2,2]:.4f}  (should be highest)")
    print(f"Gaussian kernel corner weight: {k[0,0]:.4f}  (should be lowest)")
    print(f"\nKernel:\n{np.round(k * 16).astype(int)} / 16")

    # ── Filter comparison on GAUSSIAN noise ─────────────────────────
    print("\n── Gaussian Noise Recovery (PSNR, higher=better) ──")
    results = {
        "Box 5×5":       cv2.blur(gauss_noisy, (5, 5)),
        "Gaussian 5×5":  cv2.GaussianBlur(gauss_noisy, (5, 5), 1.0),
        "Median 5×5":    cv2.medianBlur(gauss_noisy, 5),
        "Bilateral":     cv2.bilateralFilter(gauss_noisy, 9, 75, 75),
    }
    for name, result in results.items():
        print(f"  {name:20s} PSNR = {psnr(gray, result):.2f} dB")

    # ── Filter comparison on SALT & PEPPER noise ─────────────────────
    print("\n── Salt & Pepper Noise Recovery (PSNR, higher=better) ──")
    results_sp = {
        "Box 5×5":       cv2.blur(sp_noisy, (5, 5)),
        "Gaussian 5×5":  cv2.GaussianBlur(sp_noisy, (5, 5), 1.0),
        "Median 3×3":    cv2.medianBlur(sp_noisy, 3),
        "Median 5×5":    cv2.medianBlur(sp_noisy, 5),
    }
    for name, result in results_sp.items():
        print(f"  {name:20s} PSNR = {psnr(gray, result):.2f} dB")

    # ── Sigma effect demo ─────────────────────────────────────────────
    demo_sigma_effect(gray)

    print("\nDone. Key insight:")
    print("  Gaussian noise  → Gaussian blur wins")
    print("  Salt & pepper   → Median filter wins (outliers can't skew median)")


if __name__ == "__main__":
    main()
