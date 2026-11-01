# Pose Estimator — System Understanding

## What is this system?

Identifies **WHO** picked up a product from the shelf.

Product Handler tells us WHAT was picked.
Pose Estimator tells us WHO picked it.
Together → automated billing to the right customer.

---

## Full System Flow

```
IoT Sensor fires (product picked up)
      ↓
Redis (IoT DB) ← trigger stored { shelf_id, timestamp }
      ↓
Monitor Triggers (polls Redis every 100ms)
      ↓
Fetch images from RTSP camera buffer
  → Undistort fisheye lens distortion
  → Compute diff image (SSIM before vs after)
  → Apply shelf mask
      ↓
Fetch tracking details from Xovis system (Redis)
  → Xovis = external people tracking system
  → Gives: { tracking_id, coordinates, mask } per person
      ↓
Transform Xovis coordinates → camera pixel coordinates
      ↓
YOLOX Person Detector
  → Detect persons in camera frames
  → Match with Xovis tracking IDs
      ↓
Shapely Polygon Matching
  → Each shelf index = polygon in image space
  → Find which person's mask is closest to shelf polygon
  → Returns: { shelf_index: tracking_id, distance }
      ↓
Redis (Transaction DB) ← store result
  { tracking_id → transaction_id → shelf_index }
      ↓
Billing system: WHO (tracking_id) picked WHAT (Product Handler) → charge customer
```

---

## Key Components

### 1. Camera Management
```
Same as Product Handler:
  - RTSP stream via PyAV
  - One thread per camera
  - Frame buffer (Queue, max 40 frames)
  - Camera health monitor → auto-restart if unresponsive > 2 seconds
  - Fisheye undistortion on every frame
```

### 2. Fisheye Undistortion
```python
# Store cameras have wide-angle fisheye lens
# Raw frame is distorted (curved edges)
# Must undistort before person detection

K = camera matrix (intrinsics)
D = distortion coefficients
→ cv2.fisheye.initUndistortRectifyMap → cv2.remap
→ straight lines become straight again
```

### 3. Diff Image (SSIM)
```
Before image = stable frame captured when camera starts
After image  = frame after trigger fires

SSIM (Structural Similarity) compares both:
  → finds changed regions
  → filters small contours (< 10000 area)
  → dilate + erode to clean up
  → output: only changed pixels visible

Why: Focus person detection only on area where action happened
```

### 4. Xovis Tracking System
```
External hardware/software that tracks people in the store
  → gives coordinates (x, y) of each person in store space
  → gives tracking_id and transaction_id per person
  → gives mask (silhouette) of each person

Problem: Xovis uses store floor coordinates
Solution: TransformPoints converts floor coords → camera pixel coords
          (homography / calibration transformation)
```

### 5. Shelf Polygon Matching (Shapely)
```
Each shelf is divided into shelf_indexes (cells)
Each cell = polygon in image space

Calibration file defines:
  - top_left, top_right, bottom_left, bottom_right per row
  - cell_width per product

create_polygons() divides each shelf row into cell polygons

When trigger fires:
  → for each shelf_index polygon
  → for each tracked person
  → compute distance from person mask to shelf polygon
  → closest person = the one who interacted with shelf
```

### 6. YOLOX Person Detector
```
Input:  camera frame
Output: person bounding boxes in frame

Used to:
  - Confirm person is present near shelf
  - Match with Xovis tracking ID
  - Filter false triggers (no person = no valid transaction)
```

### 7. Redis DB Structure
```
REDIS_CONFIGURATION_DB    → store details, camera IDs, shelf config
REDIS_IOT_DB              → incoming triggers (IoT sensor events)
REDIS_HEALTH_DB           → camera health status
REDIS_TRANSACTION_DB      → tracking details (from Xovis), output results
```

---

## Difference from Product Handler

| | Product Handler | Pose Estimator |
|--|----------------|----------------|
| **Question** | WHAT was picked? | WHO picked it? |
| **AI Model** | SAM + ResNet | YOLOX + RTMPose |
| **Output** | product SKU + confidence | tracking_id + shelf_index |
| **Trigger** | IoT weight sensor | IoT weight sensor (same) |
| **Tracking** | Not needed | Xovis person tracking |

Both run in parallel. Billing system combines both results.

---

## Simple Flow Summary (Your Own Words)

```
Each gondola → 1 Pose camera → 2 sec buffer (continuous recording)

Trigger fires at 12:30:00
→ Take LATEST frame from buffer (person still standing at shelf)
→ YOLOX detects all persons in frame (full body boundaries)
→ Shapely checks which person is CLOSEST to shelf polygon
→ That person's body center point (pixels)
→ TransformPoints converts pixels → real-world meters
→ Match with Xovis coordinates → get Tracking ID
→ Tracking ID → Transaction ID → WHO picked product
```

### One-line Interview Answer
> "Each gondola has one top-view pose camera with a 2-second buffer. At trigger time, we take the latest frame, detect persons using YOLOX, find who is closest to the shelf using Shapely polygon matching, convert their pixel position to real-world coordinates using homography, match with Xovis tracking system to get the tracking ID, and link that to the transaction for billing."

---

## We Check BODY Position, Not Hand

Common question: "After detecting person, do you check which hand is closer to shelf?"

**No.** From top-view camera, hands are not visible reliably:
```
Top view:
      O     ← head
     [ ]    ← body (what we see)
      |
     / \
Hand goes UNDER shelf → camera can't see it
```

**Body position is enough:**
```
Person can't reach into Shelf 3 without standing in front of it.
Body position = shelf interaction zone
```

Hand tracking would need side-view cameras per shelf (like Amazon Go).

---

## How System Knows Where Shelf Is

Shelf boundaries are **defined once during store setup** — never changes unless shelf is moved.

### Two Coordinate Systems
```
Xovis floor map  → shelf position in METERS  (2.5m, 1.3m)
Camera image     → shelf position in PIXELS  (340px, 220px)
```

### Calibration Links Both
```
Store setup team marks shelf corners:
  → on floor map (meters)
  → on camera image (pixels)
  → compute homography matrix H

After calibration:
  shelf floor coords → camera pixel coords → Shapely polygon created
```

### Config File
```json
{
  "shelf_1": {
    "top_left": [100, 200],
    "top_right": [350, 200],
    "bottom_left": [100, 400],
    "bottom_right": [350, 400]
  }
}
```

### Analogy
Like drawing zones on a football field:
> Penalty box defined once → any player inside = penalty zone
> Shelf polygon defined once → any person inside = at that shelf

---

## TransformPoints — Why This Step is Critical

```
YOLOX gives person position in → PIXELS  (camera image)
Xovis gives person position in → METERS  (store floor map)

These are DIFFERENT coordinate systems → can't directly compare

TransformPoints (homography matrix H):
  pixel (340, 220) → real world (2.5m, 1.3m)

Now compare with Xovis → find closest tracking ID → WHO picked product
```

### Analogy
Like Google Maps satellite calibration:
> Satellite pixel (500, 300) = GPS (12.9716°N, 80.2209°E)
> Once calibrated → any pixel → exact real-world location

---

## Why Not MediaPipe / Hand Tracking?

Good question — why not use pose model to detect hand and find where hand came from?

### Problem 1: Camera Angle
```
MediaPipe trained on → front view / side view of person
Our camera is        → TOP VIEW (fisheye, bird's eye)

From top view person looks like:
      O        ← head (circle)
     /|\       ← body (barely visible)
     / \
→ Hand keypoints not detectable reliably from top-down
```

### Problem 2: Shelf Blocks the Hand
```
Top camera view:
[ shelf edge ][ product area ]
      ↑
   hand goes UNDER shelf → camera can't see it
```

### Problem 3: Distance
Top cameras are 3-4 meters high. Hand becomes very small → keypoint accuracy drops.

### When Hand Tracking Would Work
If side-view cameras were placed near each shelf (like Amazon Go):
> Side camera → person reaches → hand clearly visible → MediaPipe works perfectly

**Interview tip:** "If we add side-view cameras per shelf, we can use MediaPipe hand tracking for more precise pickup detection. Currently limited by top-view camera setup."

---

## How YOLOX + Shapely Works (Simple Explanation)

### Story: Security Guard Watching Top CCTV
1. Weight sensor beeps → "Something happened at Shelf 3"
2. Fisheye camera image → undistort (correct positions)
3. Compare current frame vs previous → WHERE did something move? (SSIM diff)
4. Look at that area → detect person there (YOLOX)
5. Is that person standing near Shelf 3? (Shapely polygon check)
6. That person has a tracking ID (Xovis assigns it)
7. Tracking ID → Transaction ID → bill that customer

### YOLOX gives body center point
```
YOLOX detects → full body bounding box → center point (x, y) in pixels
```

### Shapely checks: is person inside shelf polygon?
```python
shelf_polygon = Polygon([(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
person_point = Point(person_x, person_y)

shelf_polygon.contains(person_point)   # True → this person is at the shelf
shelf_polygon.distance(person_point)   # distance → find closest person
```

### Visual
```
Top view of store:

[  Shelf 1 polygon  ]    [  Shelf 2 polygon  ]

      Person A                  Person B
      center (x,y)              center (x,y)
          ↓                          ↓
    INSIDE Shelf 1              INSIDE Shelf 2

Trigger at Shelf 1 → Person A picked the product
```

---

## Calibration — Image Pixels to Real-World Coordinates

### The Problem
```
YOLOX gives       → image coordinates (pixels)
Xovis gives       → real-world coordinates (meters, floor map)

These are different coordinate systems → need to convert
```

### Solution: Homography Matrix (TransformPoints)
Calibration done once during store setup:
- Mark known points on floor
- Record their pixel position in camera image
- Compute homography matrix H

```
real_world_point = H × image_point

(340px, 220px)  →  (2.5m, 1.3m)
```

### After Calibration
```
YOLOX person point (pixels) 
→ TransformPoints → real-world (x, y) in meters
→ Match with Xovis tracking ID at same location
→ Found WHO picked the product
```

### Analogy
Like Google Maps satellite calibration:
- Satellite pixel (500, 300) = GPS (12.9716°N, 80.2209°E)
- Once calibrated → any pixel → exact real-world location

### Full Flow with Calibration
```
Trigger
→ YOLOX finds person body point (image pixels)
→ Shapely finds closest person to shelf polygon
→ TransformPoints converts pixels → real-world meters
→ Xovis has real-world (X,Y) for each tracking ID
→ Match closest Xovis coordinate → get Tracking ID
→ Tracking ID → Transaction ID → WHO picked the product
```

---

## Interview Explanation (60 seconds)

> "Pose Estimator is the customer identification component of our autonomous store. When a product pickup is detected, we need to know WHICH customer picked it for billing. The system fetches camera frames and Xovis tracking data — Xovis is an external people tracking system that gives each customer a tracking ID and their location in the store. We transform those store coordinates to camera pixel coordinates using homography. Then YOLOX detects persons in the frame. We use Shapely polygon matching — each shelf cell is defined as a polygon in image space, and we find the tracked person whose mask is geometrically closest to that polygon. That person gets linked to the transaction for billing."

---

## Resume Bullets (Digit7)

```
● Pose Estimator — Customer Identification System:
  Built real-time customer identification pipeline for autonomous
  billing — IoT trigger → RTSP camera frames → fisheye undistortion
  → SSIM diff image → YOLOX person detection → Shapely polygon
  matching to link tracked person (Xovis) to shelf interaction →
  result stored in Redis for transaction processing.
  Tools: OpenCV, YOLOX, RTMPose, Shapely, PyAV, Redis, PyTorch
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| Xovis | External people tracking hardware — gives person ID + location |
| YOLOX | Fast object detection model — detects persons in frame |
| RTMPose | Pose estimation model — detects body keypoints (currently disabled) |
| SSIM | Structural Similarity — compares two images to find changed regions |
| Shapely | Python geometry library — polygons, distance calculations |
| Fisheye undistort | Correct wide-angle camera lens distortion |
| TransformPoints | Converts Xovis store coordinates to camera pixel coordinates |
| Shelf polygon | Image-space polygon defining where each shelf cell is in camera view |
| tracking_id | Unique ID assigned by Xovis to each customer in store |
| transaction_id | Billing ID linked to tracking_id |
