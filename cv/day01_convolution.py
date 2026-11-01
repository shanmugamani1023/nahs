"""
Day 1: 2D Convolution from Scratch
Goal: Understand what cv2.filter2D actually does under the hood
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# STEP 1: Implement 2D convolution from scratch
# ─────────────────────────────────────────────

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D cross-correlation (what OpenCV/PyTorch call 'convolution').

    Formula: Output[i,j] = sum over m,n of Image[i+m, j+n] * Kernel[m, n]

    Args:
        image:  2D grayscale image (H, W), dtype float
        kernel: 2D kernel (kH, kW)
    Returns:
        output: 2D array same size as image (with SAME padding)
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # Padding to keep output same size as input (SAME padding)
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    output = np.zeros((H, W), dtype=np.float64)

    # Slide kernel over every pixel
    for i in range(H):
        for j in range(W):
            # Extract patch same size as kernel
            patch = padded[i:i + kH, j:j + kW]
            # Element-wise multiply + sum = one output pixel
            output[i, j] = np.sum(patch * kernel)

    return output


# ─────────────────────────────────────────────
# STEP 2: Vectorized version (much faster)
# ─────────────────────────────────────────────

def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Same operation but using stride tricks — no Python loops.
    This is closer to how optimized libraries work.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Use as_strided to create a view of all patches at once
    from numpy.lib.stride_tricks import as_strided
    shape = (H, W, kH, kW)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    patches = as_strided(padded, shape=shape, strides=strides)  # (H, W, kH, kW)

    # Multiply each patch by kernel, sum over last 2 dims
    output = np.einsum('ijmn,mn->ij', patches, kernel)
    return output


# ─────────────────────────────────────────────
# STEP 3: Define kernels
# ─────────────────────────────────────────────

kernels = {
    "Identity": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float64),

    "Blur (Box)": np.ones((3, 3), dtype=np.float64) / 9,

    "Gaussian Blur": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16,

    "Edge Detect": np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64),

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64),

    "Sobel X": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64),

    "Sobel Y": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64),
}


# ─────────────────────────────────────────────
# STEP 4: Load image and run
# ─────────────────────────────────────────────

def main():
    # Use a real image if available, otherwise create a synthetic one
    img_path = "test.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("No image found — creating synthetic test image")
        # Create a simple gradient + shapes image
        img_gray = np.zeros((200, 200), dtype=np.uint8)
        img_gray[50:150, 50:150] = 200          # white square
        img_gray[80:120, 80:120] = 100          # gray inner square
        img_gray += np.random.randint(0, 20, img_gray.shape, dtype=np.uint8)
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_float = img_gray.astype(np.float64)

    # ── Verify: our result vs OpenCV ──────────────────────────────────
    kernel = kernels["Gaussian Blur"]
    our_result = convolve2d_fast(image_float, kernel)
    cv2_result = cv2.filter2D(image_float, -1, kernel)

    diff = np.abs(our_result - cv2_result).max()
    print(f"Max difference vs cv2.filter2D: {diff:.6f}  ({'PASS ✓' if diff < 1.0 else 'FAIL'})")

    # ── Visualize all kernels ─────────────────────────────────────────
    n = len(kernels)
    fig, axes = plt.subplots(2, (n + 1) // 2 + 1, figsize=(16, 6))
    axes = axes.flatten()

    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')

    for idx, (name, k) in enumerate(kernels.items()):
        result = convolve2d_fast(image_float, k)
        result_display = np.clip(result, 0, 255).astype(np.uint8)
        axes[idx + 1].imshow(result_display, cmap='gray')
        axes[idx + 1].set_title(name, fontsize=9)
        axes[idx + 1].axis('off')

    for ax in axes[n + 1:]:
        ax.axis('off')

    plt.suptitle("Day 1: 2D Convolution from Scratch", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("day01_output.png", dpi=120, bbox_inches='tight')
    plt.show()
    print("Saved: day01_output.png")

    # ── Key numbers to understand ─────────────────────────────────────
    print("\n── Kernel Math Check ──")
    print(f"Gaussian kernel sums to: {kernels['Gaussian Blur'].sum():.1f}  (should be 1.0 — preserves brightness)")
    print(f"Edge kernel sums to:     {kernels['Edge Detect'].sum():.1f}  (should be 0.0 — enhances change, not absolute value)")
    print(f"Sharpen kernel sums to:  {kernels['Sharpen'].sum():.1f}  (should be 1.0 — preserves brightness + sharpens)")


if __name__ == "__main__":
    main()
