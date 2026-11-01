"""
Day 7: Feature Detection — HOG, SIFT, ORB, Feature Matching
Goal: Understand what makes a good feature and how descriptors work
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────
# STEP 1: HOG descriptor
# ─────────────────────────────────────────────

def compute_hog_manual(cell_img: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """
    Manual HOG for one 8×8 cell.
    Returns 9-bin histogram of gradient directions (0–180°).
    """
    # Compute gradients
    Gx = cv2.Sobel(cell_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    Gy = cv2.Sobel(cell_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.abs(np.arctan2(Gy, Gx) * 180 / np.pi) % 180  # 0–180°

    # Build histogram
    hist = np.zeros(n_bins)
    bin_size = 180 / n_bins

    for i in range(cell_img.shape[0]):
        for j in range(cell_img.shape[1]):
            bin_idx = int(direction[i, j] / bin_size) % n_bins
            hist[bin_idx] += magnitude[i, j]

    return hist


# ─────────────────────────────────────────────
# STEP 2: SIFT detect and describe
# ─────────────────────────────────────────────

def detect_sift(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=200)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print(f"SIFT: {len(keypoints)} keypoints, descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    return keypoints, descriptors


# ─────────────────────────────────────────────
# STEP 3: ORB detect and describe
# ─────────────────────────────────────────────

def detect_orb(gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    print(f"ORB:  {len(keypoints)} keypoints, descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    return keypoints, descriptors


# ─────────────────────────────────────────────
# STEP 4: Feature Matching
# ─────────────────────────────────────────────

def match_orb(des1, des2):
    """BFMatcher for ORB — uses Hamming distance (binary descriptors)."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def match_sift_ratio_test(des1, des2, ratio=0.75):
    """BFMatcher for SIFT — uses L2 distance + Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw_matches:
        if m.distance / n.distance < ratio:   # Lowe's ratio test
            good.append(m)
    return good


# ─────────────────────────────────────────────
# STEP 5: Run and compare
# ─────────────────────────────────────────────

def main():
    img = cv2.imread("test.jpg")
    if img is None:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[50:250, 80:320] = [180, 180, 180]
        img[100:200, 130:270] = [80, 80, 80]
        for i in range(0, 300, 20):
            cv2.line(img, (0, i), (400, i), (140,140,140), 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── HOG manual cell ──────────────────────────────────────────
    cell = gray[50:58, 80:88]   # one 8×8 cell
    hist = compute_hog_manual(cell)
    print("HOG histogram for one 8×8 cell (9 bins, 0–180°):")
    for i, val in enumerate(hist):
        print(f"  {i*20:3d}°–{(i+1)*20:3d}°: {'█' * int(val/max(hist+1e-6)*20):.20s} {val:.1f}")

    # ── OpenCV HOG ───────────────────────────────────────────────
    hog = cv2.HOGDescriptor((64, 128), (16,16), (8,8), (8,8), 9)
    resized = cv2.resize(gray, (64, 128))
    descriptor = hog.compute(resized)
    print(f"\nOpenCV HOG descriptor size: {descriptor.shape}  (should be 3780)")

    # ── SIFT ─────────────────────────────────────────────────────
    print("\n── SIFT ──")
    kp_sift, des_sift = detect_sift(gray)
    if kp_sift:
        print(f"  First keypoint: x={kp_sift[0].pt[0]:.1f}, y={kp_sift[0].pt[1]:.1f}, "
              f"scale={kp_sift[0].size:.1f}, angle={kp_sift[0].angle:.1f}°")

    # ── ORB ──────────────────────────────────────────────────────
    print("\n── ORB ──")
    kp_orb, des_orb = detect_orb(gray)
    if kp_orb:
        print(f"  First keypoint: x={kp_orb[0].pt[0]:.1f}, y={kp_orb[0].pt[1]:.1f}")
        print(f"  Descriptor dtype: {des_orb.dtype}  (uint8 binary)")

    # ── Matching: test on slightly modified image ─────────────────
    print("\n── Feature Matching ──")
    # Create second image: slight rotation
    M = cv2.getRotationMatrix2D((200, 150), 10, 1.0)
    img2 = cv2.warpAffine(img, M, (400, 300))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = cv2.ORB_create(500).detectAndCompute(gray,  None)
    kp2, des2 = cv2.ORB_create(500).detectAndCompute(gray2, None)

    if des1 is not None and des2 is not None:
        matches = match_orb(des1, des2)
        print(f"  Total ORB matches: {len(matches)}")
        print(f"  Top 5 match distances: {[round(m.distance) for m in matches[:5]]}")
        print(f"  (Lower distance = better match)")

    # ── SIFT matching with ratio test ────────────────────────────
    if des_sift is not None and len(kp_sift) > 10:
        kp1s, des1s = cv2.SIFT_create(200).detectAndCompute(gray,  None)
        kp2s, des2s = cv2.SIFT_create(200).detectAndCompute(gray2, None)
        if des2s is not None:
            good = match_sift_ratio_test(des1s, des2s)
            print(f"\n  SIFT matches after ratio test: {good.__len__()} "
                  f"(from {len(des1s)} descriptors)")


if __name__ == "__main__":
    main()
