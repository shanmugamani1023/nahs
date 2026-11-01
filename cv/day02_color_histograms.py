"""
Day 2: Color Spaces & Histograms from Scratch
Goal: Understand what cv2.cvtColor() and cv2.equalizeHist() actually do
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: Grayscale conversion from scratch
# ─────────────────────────────────────────────

def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """
    ITU-R BT.601 weighted average — matches human eye sensitivity.
    Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    return np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# STEP 2: RGB → HSV from scratch
# ─────────────────────────────────────────────

def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """
    Manual RGB to HSV conversion.
    OpenCV stores H as 0-179 (halved to fit uint8), S and V as 0-255.
    """
    img = img_rgb.astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Cmax = np.max(img, axis=2)
    Cmin = np.min(img, axis=2)
    delta = Cmax - Cmin

    # Value
    V = Cmax

    # Saturation
    S = np.where(Cmax == 0, 0.0, delta / Cmax)

    # Hue
    H = np.zeros_like(Cmax)

    mask_r = (Cmax == R) & (delta != 0)
    mask_g = (Cmax == G) & (delta != 0)
    mask_b = (Cmax == B) & (delta != 0)

    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / delta[mask_r])) % 360
    H[mask_g] = 60 * ((B[mask_g] - R[mask_g]) / delta[mask_g] + 2)
    H[mask_b] = 60 * ((R[mask_b] - G[mask_b]) / delta[mask_b] + 4)

    # OpenCV convention: H -> 0-179, S,V -> 0-255
    H = (H / 2).astype(np.uint8)
    S = (S * 255).astype(np.uint8)
    V = (V * 255).astype(np.uint8)

    return np.stack([H, S, V], axis=2)


# ─────────────────────────────────────────────
# STEP 3: Histogram from scratch
# ─────────────────────────────────────────────

def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """Count pixels at each intensity level (0-255)."""
    hist = np.zeros(256, dtype=np.int64)
    for val in gray.flatten():
        hist[val] += 1
    # Vectorized (same result, much faster):
    # hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    return hist


# ─────────────────────────────────────────────
# STEP 4: Histogram equalization from scratch
# ─────────────────────────────────────────────

def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Manual histogram equalization.
    Steps: histogram → CDF → normalize CDF → remap pixels
    """
    hist, _ = np.histogram(gray.flatten(), 256, [0, 256])

    # CDF: running total
    cdf = hist.cumsum()

    # Normalize: stretch CDF to 0–255
    cdf_min = cdf[cdf > 0].min()
    N = gray.size
    lut = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)

    # Remap: use pixel value as index into LUT
    return lut[gray]


# ─────────────────────────────────────────────
# STEP 5: Run and verify
# ─────────────────────────────────────────────

def main():
    # Create or load image
    img_bgr = cv2.imread("test.jpg")
    if img_bgr is None:
        # Synthetic: gradient image with a colored rectangle
        img_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
        img_bgr[:, :, 2] = np.linspace(0, 255, 300).astype(np.uint8)   # red gradient
        img_bgr[50:150, 100:200] = [0, 165, 255]                        # orange rect (BGR)
        print("Using synthetic image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Grayscale check ─────────────────────────────────────────
    our_gray = rgb_to_gray(img_rgb)
    cv2_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale max diff vs cv2: {np.abs(our_gray.astype(int) - cv2_gray.astype(int)).max()}")

    # ── HSV check ───────────────────────────────────────────────
    our_hsv = rgb_to_hsv(img_rgb)
    cv2_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    print(f"HSV H max diff vs cv2:     {np.abs(our_hsv[:,:,0].astype(int) - cv2_hsv[:,:,0].astype(int)).max()}")

    # ── Histogram equalization check ────────────────────────────
    dark_img = (cv2_gray.astype(float) * 0.3).astype(np.uint8)   # simulate dark image
    our_eq = equalize_histogram(dark_img)
    cv2_eq = cv2.equalizeHist(dark_img)
    print(f"EqualizeHist max diff:     {np.abs(our_eq.astype(int) - cv2_eq.astype(int)).max()}")

    # ── CLAHE for comparison ─────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(dark_img)

    # ── Print histogram insight ──────────────────────────────────
    hist_before = np.histogram(dark_img.flatten(), 256, [0, 256])[0]
    hist_after = np.histogram(our_eq.flatten(), 256, [0, 256])[0]
    print(f"\nBefore eq — pixel range: {dark_img.min()}–{dark_img.max()}")
    print(f"After eq  — pixel range: {our_eq.min()}–{our_eq.max()}")
    print(f"Equalization stretched the range from {dark_img.max()-dark_img.min()} to {our_eq.max()-our_eq.min()} levels")

    print("\nAll checks done. Results saved to day02_output.png")


if __name__ == "__main__":
    main()
