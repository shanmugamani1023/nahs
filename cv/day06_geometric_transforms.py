"""
Day 6: Geometric Transforms — Affine, Perspective, Homography
Goal: Understand what warpAffine and warpPerspective do mathematically
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: Affine transform from scratch
# ─────────────────────────────────────────────

def apply_affine_manual(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply 2×3 affine matrix manually.
    For each output pixel (x',y'), find source pixel (x,y) using inverse.
    Formula: [x',y'] = M × [x,y,1]
    """
    H, W = img.shape[:2]
    output = np.zeros_like(img)

    # Inverse mapping: for each output pixel, find where it came from
    M_inv = cv2.invertAffineTransform(M)

    for y_out in range(H):
        for x_out in range(W):
            # Apply inverse affine to find source coordinates
            src = M_inv @ np.array([x_out, y_out, 1.0])
            x_src, y_src = int(round(src[0])), int(round(src[1]))

            # Copy pixel if source is within bounds
            if 0 <= x_src < W and 0 <= y_src < H:
                output[y_out, x_out] = img[y_src, x_src]

    return output


# ─────────────────────────────────────────────
# STEP 2: Perspective transform — document scanner
# ─────────────────────────────────────────────

def document_scanner(img: np.ndarray, corners: np.ndarray, output_size=(400, 600)):
    """
    Perspective transform: correct skewed document to flat rectangle.
    corners: 4 points [top-left, top-right, bottom-right, bottom-left]
    """
    W, H = output_size
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    M      = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    result = cv2.warpPerspective(img, M, (W, H))
    return result, M


# ─────────────────────────────────────────────
# STEP 3: Common affine operations
# ─────────────────────────────────────────────

def rotate_image(img, angle_deg, scale=1.0):
    H, W = img.shape[:2]
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img, M, (W, H))

def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    H, W = img.shape[:2]
    return cv2.warpAffine(img, M, (W, H))

def shear_image(img, shear_x=0.2):
    H, W = img.shape[:2]
    M = np.float32([[1, shear_x, 0],
                    [0, 1,       0]])
    return cv2.warpAffine(img, M, (W, H))


# ─────────────────────────────────────────────
# STEP 4: Run and verify
# ─────────────────────────────────────────────

def main():
    img = cv2.imread("test.jpg")
    if img is None:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[50:250, 80:320] = [200, 200, 200]
        cv2.putText(img, "DAY 6", (130, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (50, 50, 200), 3)

    H, W = img.shape[:2]

    # ── Affine: Rotation ─────────────────────────────────────────
    rotated = rotate_image(img, 30)
    print(f"Rotation: image shape unchanged {rotated.shape}")

    # ── Affine: what getAffineTransform gives ────────────────────
    src = np.float32([[0,0], [W-1,0], [0,H-1]])
    dst = np.float32([[0,10], [W-1,0], [10,H-1]])
    M_affine = cv2.getAffineTransform(src, dst)
    print(f"\nAffine matrix (2×3):\n{M_affine}")

    # ── Perspective: simulate document scan ──────────────────────
    # Simulate a skewed document (trapezoid)
    margin = 50
    corners = np.float32([
        [margin,   margin],           # top-left
        [W-margin, margin//2],        # top-right (slightly higher = skew)
        [W-margin//2, H-margin//2],   # bottom-right
        [margin//2, H-margin],        # bottom-left
    ])
    scanned, M_persp = document_scanner(img, corners, output_size=(W, H))
    print(f"\nPerspective matrix (3×3):\n{np.round(M_persp, 3)}")

    # ── Verify: affine preserves parallel lines ──────────────────
    print("\n── Affine vs Perspective — key difference ──")
    print("Affine  (2×3): parallel lines → ALWAYS stay parallel")
    print("Persp   (3×3): parallel lines → may converge (vanishing point)")
    print("Homography   : same as perspective but computed from feature matches")

    # ── Homography from manual points ────────────────────────────
    src_4 = np.float32([[0,0],[W,0],[W,H],[0,H]])
    dst_4 = np.float32([[10,20],[W-10,15],[W-20,H-10],[15,H-20]])
    H_mat, _ = cv2.findHomography(src_4, dst_4)
    print(f"\nHomography matrix (3×3):\n{np.round(H_mat, 3)}")

    # ── Data augmentation example ─────────────────────────────────
    print("\nData augmentation using affine (common in CV training):")
    for angle in [0, 15, 30, 45]:
        aug = rotate_image(img, angle)
        print(f"  Rotated {angle}° → shape: {aug.shape}")


if __name__ == "__main__":
    main()
