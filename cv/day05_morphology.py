"""
Day 5: Morphological Operations from scratch
Goal: Understand erosion, dilation, opening, closing — what they do pixel by pixel
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: Erosion from scratch
# ─────────────────────────────────────────────

def erode(binary_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Erosion: for each pixel, if ALL pixels under kernel = 255 → keep 255
                                  if ANY pixel under kernel = 0   → set to 0
    Effect: white regions shrink, small objects disappear
    """
    H, W = binary_img.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    padded = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    output = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            patch = padded[i:i+kH, j:j+kW]
            # ALL pixels under kernel must be 255 (where kernel=1)
            region = patch[kernel == 1]
            output[i, j] = 255 if np.all(region == 255) else 0

    return output


# ─────────────────────────────────────────────
# STEP 2: Dilation from scratch
# ─────────────────────────────────────────────

def dilate(binary_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Dilation: for each pixel, if ANY pixel under kernel = 255 → set to 255
                                   if ALL pixels = 0           → keep 0
    Effect: white regions expand, small holes get filled
    """
    H, W = binary_img.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    padded = np.pad(binary_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            patch = padded[i:i+kH, j:j+kW]
            region = patch[kernel == 1]
            output[i, j] = 255 if np.any(region == 255) else 0

    return output


# ─────────────────────────────────────────────
# STEP 3: Opening and Closing
# ─────────────────────────────────────────────

def opening(binary_img, kernel):
    """Erosion → Dilation. Removes small white noise."""
    return dilate(erode(binary_img, kernel), kernel)

def closing(binary_img, kernel):
    """Dilation → Erosion. Fills small black holes."""
    return erode(dilate(binary_img, kernel), kernel)


# ─────────────────────────────────────────────
# STEP 4: Create test binary images
# ─────────────────────────────────────────────

def create_noisy_binary():
    """Large white object + small noise dots."""
    img = np.zeros((100, 150), dtype=np.uint8)
    img[30:70, 40:110] = 255          # large white rectangle
    # Add small noise dots
    img[10, 20]  = 255
    img[15, 130] = 255
    img[80, 50]  = 255
    img[85, 100] = 255
    img[5, 75]   = 255
    return img

def create_holey_binary():
    """White object with small black holes inside."""
    img = np.zeros((100, 150), dtype=np.uint8)
    img[20:80, 30:120] = 255          # large white rectangle
    # Add small black holes
    img[35, 60]  = 0
    img[40, 90]  = 0
    img[55, 70]  = 0
    img[60, 100] = 0
    img[45, 50]  = 0
    return img


# ─────────────────────────────────────────────
# STEP 5: Run and verify
# ─────────────────────────────────────────────

def main():
    kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
    kernel_5x5 = np.ones((5, 5), dtype=np.uint8)

    noisy  = create_noisy_binary()
    holey  = create_holey_binary()

    # ── Erosion: shrinks objects ─────────────────────────────────
    eroded = erode(noisy, kernel_3x3)
    cv2_eroded = cv2.erode(noisy, kernel_3x3)
    print(f"Erosion match vs cv2: {np.array_equal(eroded, cv2_eroded)}")

    # ── Dilation: grows objects ──────────────────────────────────
    dilated = dilate(noisy, kernel_3x3)
    cv2_dilated = cv2.dilate(noisy, kernel_3x3)
    print(f"Dilation match vs cv2: {np.array_equal(dilated, cv2_dilated)}")

    # ── Opening: removes noise dots ──────────────────────────────
    noise_dots_before = np.sum(noisy == 255)
    opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel_3x3)
    noise_dots_after = np.sum(opened == 255)
    print(f"\nOpening (noise removal):")
    print(f"  White pixels before: {noise_dots_before}")
    print(f"  White pixels after:  {noise_dots_after}")
    print(f"  Removed:             {noise_dots_before - noise_dots_after} pixels (noise dots)")

    # ── Closing: fills holes ─────────────────────────────────────
    holes_before = np.sum(holey[20:80, 30:120] == 0)
    closed = cv2.morphologyEx(holey, cv2.MORPH_CLOSE, kernel_3x3)
    holes_after = np.sum(closed[20:80, 30:120] == 0)
    print(f"\nClosing (hole filling):")
    print(f"  Black pixels (holes) before: {holes_before}")
    print(f"  Black pixels (holes) after:  {holes_after}")
    print(f"  Filled:                      {holes_before - holes_after} pixels")

    # ── Morphological gradient: boundary ─────────────────────────
    gradient = cv2.morphologyEx(noisy, cv2.MORPH_GRADIENT, kernel_3x3)
    print(f"\nGradient (boundary) non-zero pixels: {np.count_nonzero(gradient)}")

    # ── Effect of kernel size ─────────────────────────────────────
    print("\nKernel size effect on erosion (white pixels remaining):")
    for k in [3, 5, 7, 9]:
        kern = np.ones((k, k), dtype=np.uint8)
        result = cv2.erode(noisy, kern)
        print(f"  {k}×{k} kernel → {np.sum(result==255)} white pixels")
    print("  (Larger kernel = more erosion = fewer white pixels)")

    # ── Real pipeline: defect detection ──────────────────────────
    print("\nTypical defect detection pipeline:")
    img = cv2.imread("test.jpg")
    if img is None:
        test = np.random.randint(100, 150, (200, 300), dtype=np.uint8)
        test[80:90, 100:120] = 220     # simulate defect
        test[50, 60] = 230             # simulate noise dot
        gray = test
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove noise
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, kernel)  # fill holes
    print(f"  Before pipeline: {np.count_nonzero(binary)} white pixels")
    print(f"  After pipeline:  {np.count_nonzero(clean)} white pixels")


if __name__ == "__main__":
    main()
