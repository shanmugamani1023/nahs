"""
Day 4: Edge Detection — Sobel gradients + Canny pipeline from scratch
Goal: Understand what cv2.Canny() does step by step internally
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: Sobel gradients from scratch
# ─────────────────────────────────────────────

def sobel_gradients(image: np.ndarray):
    """
    Compute Gx, Gy, magnitude, direction using Sobel kernels.
    """
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    # Convolve (using OpenCV for speed, concept is same as Day 1)
    Gx = cv2.filter2D(image.astype(np.float64), -1, sobel_x)
    Gy = cv2.filter2D(image.astype(np.float64), -1, sobel_y)

    magnitude  = np.sqrt(Gx**2 + Gy**2)
    direction  = np.arctan2(Gy, Gx) * 180 / np.pi   # degrees

    return Gx, Gy, magnitude, direction


# ─────────────────────────────────────────────
# STEP 2: Non-Maximum Suppression from scratch
# ─────────────────────────────────────────────

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Thin edges to 1 pixel width.
    For each pixel: check 2 neighbours along gradient direction.
    Keep only if it's the local maximum.
    """
    H, W = magnitude.shape
    result = np.zeros((H, W), dtype=np.float64)

    # Normalize direction to 0, 45, 90, 135 degrees
    angle = direction % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q, r = 255, 255   # neighbours to compare

            # 0°  → horizontal edge → compare left/right
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # 45° → diagonal edge
            elif 22.5 <= angle[i,j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # 90° → vertical edge → compare top/bottom
            elif 67.5 <= angle[i,j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # 135° → diagonal edge
            elif 112.5 <= angle[i,j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # Keep only if this pixel is the local maximum
            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                result[i,j] = magnitude[i,j]

    return result


# ─────────────────────────────────────────────
# STEP 3: Double threshold + Hysteresis
# ─────────────────────────────────────────────

def double_threshold(image: np.ndarray, low: float, high: float):
    """
    Classify pixels as strong, weak, or suppressed.
    """
    strong = 255
    weak   = 75

    result = np.zeros_like(image, dtype=np.uint8)
    result[image >= high] = strong          # definitely edge
    result[(image >= low) & (image < high)] = weak   # maybe edge
    # below low → stays 0 (noise, discard)

    return result, strong, weak


def hysteresis(image: np.ndarray, strong: int = 255, weak: int = 75) -> np.ndarray:
    """
    Keep weak edges only if they are connected to a strong edge.
    """
    H, W = image.shape
    result = image.copy()

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if result[i,j] == weak:
                # Check if any 8-connected neighbour is strong
                neighbourhood = result[i-1:i+2, j-1:j+2]
                if strong in neighbourhood:
                    result[i,j] = strong   # connected to real edge → keep
                else:
                    result[i,j] = 0        # isolated weak → discard

    return result


# ─────────────────────────────────────────────
# STEP 4: Full Canny from scratch
# ─────────────────────────────────────────────

def canny_from_scratch(gray: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Full Canny edge detector:
    1. Gaussian blur
    2. Sobel gradients
    3. Gradient magnitude + direction
    4. Non-maximum suppression
    5. Double threshold + hysteresis
    """
    # Step 1: Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Step 2+3: Gradients
    _, _, magnitude, direction = sobel_gradients(blur)

    # Step 4: NMS
    thin = non_maximum_suppression(magnitude, direction)

    # Step 5: Double threshold + hysteresis
    thresh, strong, weak = double_threshold(thin, low_threshold, high_threshold)
    edges = hysteresis(thresh, strong, weak)

    return edges


# ─────────────────────────────────────────────
# STEP 5: Run and compare
# ─────────────────────────────────────────────

def main():
    img = cv2.imread("test.jpg")
    if img is None:
        print("Creating synthetic test image")
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[40:160, 40:260] = 200
        img[70:130, 100:200] = 80

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Our Canny vs OpenCV Canny
    our_edges = canny_from_scratch(gray, low_threshold=50, high_threshold=150)
    cv2_edges = cv2.Canny(gray, 50, 150)

    # Sobel gradients visualization
    Gx, Gy, magnitude, direction = sobel_gradients(gray.astype(np.float64))
    mag_norm = np.clip(magnitude, 0, 255).astype(np.uint8)

    print("Canny from scratch vs OpenCV:")
    print(f"  Our edges  — non-zero pixels: {np.count_nonzero(our_edges)}")
    print(f"  cv2.Canny  — non-zero pixels: {np.count_nonzero(cv2_edges)}")

    print("\nGradient stats:")
    print(f"  Max Gx magnitude: {np.abs(Gx).max():.1f}")
    print(f"  Max Gy magnitude: {np.abs(Gy).max():.1f}")
    print(f"  Max combined magnitude: {magnitude.max():.1f}")

    print("\nCanny threshold effect:")
    for low, high in [(20, 60), (50, 150), (100, 200)]:
        edges = cv2.Canny(gray, low, high)
        print(f"  T_low={low:3d}, T_high={high:3d} → edge pixels: {np.count_nonzero(edges)}")
    print("  (Lower thresholds = more edges detected, including noise)")


if __name__ == "__main__":
    main()
