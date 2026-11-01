# Computer Vision — Question Bank
## From Scratch to Current State of the Art

---

# SECTION 1: IMAGE BASICS

**Q1. What is a digital image?**
> A 2D grid of pixels. Each pixel has intensity value.
> Grayscale: 1 channel (0-255). Color: 3 channels (R, G, B).
> Image shape = (Height, Width, Channels)

**Q2. What is the difference between RGB and BGR?**
> RGB → Red, Green, Blue (standard, used by matplotlib, PIL)
> BGR → Blue, Green, Red (used by OpenCV by default)
> Always convert: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before displaying

**Q3. What is image resolution?**
> Number of pixels in width × height.
> Higher resolution = more detail = larger file size.
> 1080p = 1920×1080 pixels

**Q4. What is a pixel value and what does it represent?**
> Single number (0-255) for grayscale.
> 0 = black, 255 = white.
> Color pixel = (R, G, B) tuple.

**Q5. What is image normalization? Why do we do it?**
> Scale pixel values from [0, 255] to [0, 1] or [-1, 1].
> `img = img / 255.0`
> Why: Neural networks train faster and more stably with small values.
> Large values cause gradient explosion.

**Q6. What is the difference between image and video?**
> Video = sequence of images (frames) played at FPS (frames per second).
> 30 FPS = 30 images shown per second = appears smooth to human eye.

**Q7. What is FPS and why does it matter in CV?**
> FPS = Frames Per Second.
> Your CV model must process faster than incoming FPS.
> If camera = 30 FPS and model = 10 FPS → you fall behind → lag.
> Real-time CV requires model FPS ≥ camera FPS.

---

# SECTION 2: IMAGE PROCESSING (OpenCV)

**Q8. What is grayscale conversion and why use it?**
> Convert 3-channel color to 1-channel.
> `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
> Why: Reduces computation by 3x. Many algorithms work on single channel.

**Q9. What is image thresholding?**
> Convert grayscale to binary (0 or 255) based on threshold value.
> `cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)`
> Pixels > 127 → 255 (white). Pixels ≤ 127 → 0 (black).
> Used to separate foreground from background.

**Q10. What is Gaussian blur and why use it?**
> Smoothing filter — reduces noise by averaging nearby pixels.
> `cv2.GaussianBlur(img, (5,5), 0)`
> Why: Remove noise before edge detection or thresholding.
> Larger kernel = more blur.

**Q11. What is edge detection? How does Canny work?**
> Finding boundaries between regions in an image.
> Canny steps:
>   1. Gaussian blur (remove noise)
>   2. Gradient calculation (find intensity changes)
>   3. Non-max suppression (thin edges)
>   4. Double threshold (strong/weak edges)
>   5. Hysteresis (keep connected strong edges)
> `cv2.Canny(img, 100, 200)`

**Q12. What is morphological operation? Erosion vs Dilation?**
> Operations on binary images to change shape of regions.
> Erosion: shrinks white regions (removes small noise)
> Dilation: grows white regions (fills small holes)
> `cv2.erode(img, kernel)` / `cv2.dilate(img, kernel)`
> Used after thresholding to clean up masks.

**Q13. What is contour detection?**
> Find boundaries of connected white regions in binary image.
> `cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
> Returns list of contours (each = array of boundary points).
> Use `cv2.boundingRect(contour)` to get bounding box.

**Q14. What is cv2.absdiff?**
> Absolute difference between two images pixel by pixel.
> `diff = cv2.absdiff(frame1, frame2)`
> Result: bright where pixels changed, dark where same.
> Used for motion detection and change detection.

**Q15. What is SSIM?**
> Structural Similarity Index — compares two images for similarity.
> Returns score 0 to 1 (1 = identical).
> Better than pixel difference — considers structure, not just intensity.
> `from skimage.metrics import structural_similarity as ssim`
> Used in Pose Estimator to find active region between frames.

**Q16. What is image resizing? What interpolation methods exist?**
> Change image dimensions.
> `cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)`
> Interpolation methods:
>   INTER_NEAREST  → fastest, blocky
>   INTER_LINEAR   → good quality, default
>   INTER_CUBIC    → better quality, slower
>   INTER_AREA     → best for shrinking

**Q17. What is image augmentation? Why is it used?**
> Artificially increase training data by transforming images:
>   - Flip (horizontal/vertical)
>   - Rotation (±10°)
>   - Brightness change
>   - Crop and resize
>   - Noise addition
> Why: Prevents overfitting. Model learns to handle real-world variations.

---

# SECTION 3: CLASSICAL COMPUTER VISION

**Q18. What is HOG (Histogram of Oriented Gradients)?**
> Feature descriptor — describes local shape/texture.
> Divide image into cells → compute gradient direction histogram per cell.
> Used in pedestrian detection before deep learning.
> Limitation: Not robust to scale/rotation changes.

**Q19. What is SIFT? Why was it important?**
> Scale-Invariant Feature Transform.
> Detects keypoints that are stable across scale and rotation.
> Used for image matching, panorama stitching, object recognition.
> Limitation: Slow, patented (now free). Replaced by deep features in most cases.

**Q20. What is ORB?**
> Oriented FAST and Rotated BRIEF.
> Fast alternative to SIFT/SURF — free and real-time capable.
> Used for feature matching in AR and robotics.
> `cv2.ORB_create()`

**Q21. What is optical flow?**
> Track pixel/feature motion between consecutive frames.
> Lucas-Kanade: sparse optical flow (track specific points)
> `cv2.calcOpticalFlowPyrLK(prev, curr, points, None)`
> Used for: motion estimation, video stabilization, tracking.

**Q22. What is background subtraction? How does MOG2 work?**
> Separate foreground (moving objects) from background (static scene).
> MOG2 = Mixture of Gaussians v2:
>   - Models each pixel as mixture of Gaussian distributions
>   - Background = dominant Gaussian (stable over time)
>   - Foreground = pixel that doesn't fit background model
>   - Returns: foreground mask (white = motion, black = background)
> `mog = cv2.createBackgroundSubtractorMOG2()`
> `mask = mog.apply(frame)`
> Motion score = count of white pixels in mask.

**Q23. What is homography?**
> 3×3 matrix that maps points from one 2D plane to another.
> Used for:
>   - Image stitching (panorama)
>   - Perspective correction
>   - Coordinate transformation (pixel → real-world)
> `H, _ = cv2.findHomography(src_points, dst_points)`
> `transformed = cv2.perspectiveTransform(points, H)`

**Q24. What is camera calibration?**
> Find camera's intrinsic parameters (focal length, principal point, distortion).
> Done by photographing a chessboard pattern from multiple angles.
> Output: camera matrix K + distortion coefficients D.
> Required before: undistortion, stereo depth, AR applications.

**Q25. What is fisheye distortion and how to correct it?**
> Fisheye lens captures wide angle but distorts image (curved lines).
> Correction: undistortion using calibration parameters.
> `cv2.fisheye.initUndistortRectifyMap(K, D, ...)`
> `cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)`
> Used in Pose Estimator — top-view store cameras are fisheye.

**Q26. What is stereo vision? How is depth calculated?**
> Two cameras side by side (like human eyes).
> Disparity = horizontal difference in where same point appears in left vs right image.
> Depth = (focal_length × baseline) / disparity
> Baseline = distance between two cameras.
> Used in: Xovis tracking, autonomous vehicles, robotics.

---

# SECTION 4: DEEP LEARNING FUNDAMENTALS FOR CV

**Q27. What is a Convolutional Neural Network (CNN)?**
> Neural network designed for images.
> Key layers:
>   Convolution → extract features (edges, textures, shapes)
>   Pooling     → reduce spatial size (downsample)
>   ReLU        → add non-linearity
>   Flatten     → convert 2D to 1D
>   FC Layer    → final classification

**Q28. What does a convolution operation do?**
> Slide a small filter (kernel) over image → compute dot product at each position.
> Result: feature map showing where that pattern exists.
> 3×3 filter: detects local patterns (edges, corners).
> Multiple filters → multiple feature maps → rich feature representation.

**Q29. What is pooling? Max pooling vs Average pooling?**
> Reduce spatial dimensions while keeping important features.
> Max pooling: take maximum value in each region → keeps strongest feature.
> Average pooling: take average → smoother representation.
> Most common: 2×2 max pooling → halves width and height.

**Q30. What is stride and padding in convolution?**
> Stride: how many pixels to move filter each step.
>   Stride 1 → output same size (approximately)
>   Stride 2 → output half size (downsampling)
> Padding: add border of zeros around image.
>   'same' padding → output same spatial size as input
>   'valid' padding → no padding → output shrinks

**Q31. What is batch normalization and why use it?**
> Normalize activations within each mini-batch.
> Why: Prevents internal covariate shift → faster training, more stable.
> Applied after convolution, before activation (usually).
> Effect: Can use higher learning rates, acts as regularizer.

**Q32. What is dropout and why use it?**
> Randomly set fraction of neurons to zero during training.
> `Dropout(0.5)` → 50% of neurons dropped each step.
> Why: Prevents overfitting — model can't rely on any one neuron.
> Disabled during inference (test time).

**Q33. What is transfer learning? Why is it important for CV?**
> Use weights from model trained on large dataset (ImageNet) as starting point.
> Fine-tune on your smaller dataset.
> Why:
>   - ImageNet pretrained models already know edges, textures, shapes
>   - You only need to teach the specific task
>   - Need much less data and training time
> Example: Take ResNet50 → replace last layer → train on your 300 product images.

**Q34. What is overfitting? How to detect and prevent it?**
> Model memorizes training data → fails on new data.
> Detection: Training accuracy high, validation accuracy low.
> Prevention:
>   - More data (or augmentation)
>   - Dropout
>   - Batch normalization
>   - Early stopping
>   - Smaller model
>   - Regularization (L1/L2)

**Q35. What is data imbalance? How to handle it in CV?**
> Some classes have far more samples than others.
> Problems: Model biased toward majority class.
> Solutions:
>   - Oversample minority class (SMOTE for images = augmentation)
>   - Undersample majority class
>   - Class weights in loss function
>   - Focal loss (penalizes easy examples more)

---

# SECTION 5: CNN ARCHITECTURES

**Q36. What is VGG? Why was it significant?**
> VGG16/VGG19 — very deep network with 3×3 convolutions only.
> Significance: Showed that depth matters — deeper = better.
> Limitation: Very large (138M parameters) — slow inference.

**Q37. What is ResNet? What problem does it solve?**
> Residual Network — introduces skip connections.
> Problem solved: Vanishing gradient in very deep networks.
> Skip connection: output = F(x) + x (add input to output)
> Effect: Gradients flow directly through skip connection → can train 100+ layers.
> WideResNet50: wider channels instead of more depth → faster.

**Q38. What is EfficientNet?**
> Systematically scales width, depth, resolution together (compound scaling).
> Result: Same accuracy as larger models with fewer parameters.
> EfficientNet-B0 to B7 — progressively larger.
> Used when you need accuracy + efficiency.

**Q39. What is MobileNet? When to use it?**
> Lightweight CNN designed for mobile/edge devices.
> Uses depthwise separable convolutions → 8-9x fewer operations.
> Trade-off: Slightly lower accuracy, much faster inference.
> Use when: Edge device, real-time constraint, limited GPU.

**Q40. What is WideResNet? Why was it used in Product Handler?**
> ResNet variant that increases channel width instead of depth.
> WideResNet50: same depth as ResNet50 but wider layers.
> Trade-off: More parameters than ResNet50 but better accuracy.
> Used in Product Handler: Product classification needs high accuracy (95%+ target).

---

# SECTION 6: OBJECT DETECTION

**Q41. What is object detection vs image classification?**
> Classification: What is in the image? → one label for whole image.
> Detection: What is in the image AND where? → label + bounding box for each object.

**Q42. What is a bounding box? How is it represented?**
> Rectangle around detected object.
> Format 1: [x_min, y_min, x_max, y_max] (top-left + bottom-right corners)
> Format 2: [x_center, y_center, width, height] (YOLO format)
> Confidence score: how sure the model is (0-1).

**Q43. What is IoU (Intersection over Union)?**
> Measures overlap between predicted and ground truth bounding box.
> IoU = Area of Overlap / Area of Union
> IoU = 1.0 → perfect match
> IoU = 0.0 → no overlap
> Threshold: typically IoU > 0.5 = correct detection.

**Q44. What is NMS (Non-Maximum Suppression)?**
> Remove duplicate detections of same object.
> Steps:
>   1. Sort detections by confidence score
>   2. Keep highest confidence detection
>   3. Remove all others with IoU > threshold (0.5)
>   4. Repeat for remaining detections
> Without NMS: same object detected 10 times → noisy output.

**Q45. What is mAP (Mean Average Precision)?**
> Standard metric for evaluating object detectors.
> AP = area under precision-recall curve for one class.
> mAP = average AP across all classes.
> mAP@50 = mAP at IoU threshold 0.5.
> mAP@50:95 = average mAP across IoU thresholds 0.5 to 0.95.

**Q46. What is YOLO? How does it work?**
> You Only Look Once — real-time object detector.
> Single pass through network → directly predicts boxes + classes.
> Divides image into grid → each cell predicts N boxes.
> Very fast (real-time) vs two-stage detectors (Faster RCNN).
> Versions: YOLOv1 → YOLOv8/v11 (getting better each version).

**Q47. What is YOLOX?**
> Improved YOLO variant — anchor-free detector.
> Anchor-free: no predefined box sizes needed.
> Decoupled head: separate heads for classification and regression.
> Better small object detection than original YOLO.
> Used in Pose Estimator for person detection.

**Q48. What is the difference between one-stage and two-stage detectors?**
```
Two-stage (Faster RCNN):
  Stage 1: Region Proposal Network (RPN) → propose candidate regions
  Stage 2: Classify and refine each region
  → High accuracy, slow

One-stage (YOLO, SSD):
  Single network → directly predict boxes + classes
  → Faster, slightly lower accuracy
  → Better for real-time applications
```

**Q49. What is anchor box? What is anchor-free detection?**
> Anchor box: predefined box sizes/ratios the model starts from.
> Model predicts offsets from anchors → final box.
> Problem with anchors: Need to design anchor sizes manually.
> Anchor-free: Predict box directly from points — no predefined sizes needed.
> YOLOX, FCOS, CenterNet are anchor-free.

**Q50. How do you handle small object detection?**
> Small objects are hard — few pixels → little information.
> Solutions:
>   - Higher resolution input images
>   - FPN (Feature Pyramid Network) — detect at multiple scales
>   - Smaller anchor sizes
>   - Tile the image (split into overlapping crops, detect on each)
>   - Data augmentation with small objects

---

# SECTION 7: IMAGE SEGMENTATION

**Q51. What are the types of segmentation?**
```
Semantic segmentation:
  → Label every pixel with a class
  → All persons = same color
  → No distinction between instances

Instance segmentation:
  → Label every pixel + distinguish individual objects
  → Person 1 = red, Person 2 = blue
  → Mask RCNN is classic example

Panoptic segmentation:
  → Combines both: semantic + instance
  → Every pixel labeled, every instance separated
```

**Q52. What is Mask RCNN?**
> Extends Faster RCNN with a mask prediction branch.
> For each detected object: outputs bounding box + class + pixel mask.
> First major instance segmentation model.
> Limitation: Slow for real-time applications.

**Q53. What is SAM (Segment Anything Model)?**
> Meta's foundation model for segmentation.
> Trained on 1 billion masks — largest segmentation dataset ever.
> Accepts prompts:
>   - Point → "segment what's here"
>   - Bounding box → "segment inside this box"
>   - Text → "segment the red shoe"
> Output: precise pixel mask.
> Why used in Product Handler: handles irregular product shapes better than YOLO segments or raw bounding boxes. No retraining needed for new products.

**Q54. What is SAM2?**
> SAM extended to videos.
> Track and segment objects across video frames.
> Memory mechanism: remembers object from previous frames.
> Use case: Video segmentation, object tracking with masks.

**Q55. What is semantic segmentation used for in industry?**
> Autonomous driving: segment road, pedestrians, cars, signs.
> Medical imaging: segment tumor, organ boundaries.
> Satellite imagery: segment buildings, roads, forests.
> Retail: segment shelf regions, products.

**Q56. What is U-Net?**
> Encoder-decoder architecture for semantic segmentation.
> Encoder: CNN that downsamples → extracts features.
> Decoder: Upsamples back to original resolution.
> Skip connections: connect encoder to decoder at each scale.
> Originally designed for medical image segmentation.
> Still widely used due to simplicity and effectiveness.

---

# SECTION 8: POSE ESTIMATION

**Q57. What is pose estimation?**
> Detect body keypoints (joints) of a person.
> Keypoints: head, shoulders, elbows, wrists, hips, knees, ankles.
> Output: 17-133 keypoints with (x, y, confidence) each.
> Used for: action recognition, sports analysis, gesture control, person tracking.

**Q58. What is the difference between top-down and bottom-up pose estimation?**
```
Top-down:
  Step 1: Detect all persons (YOLO/Faster RCNN)
  Step 2: For each person, estimate pose
  → High accuracy, slow for many people (runs pose model N times)

Bottom-up:
  Step 1: Detect all keypoints in image at once
  Step 2: Group keypoints to persons
  → Faster, less accurate for crowded scenes
  → OpenPose uses bottom-up
```

**Q59. What is MediaPipe?**
> Google's framework for real-time CV pipelines.
> Includes: Hand detection, face mesh, body pose, object detection.
> Very fast — optimized for mobile/edge.
> Limitation: Works best with front/side view. Struggles with top-down view.

**Q60. What is RTMPose?**
> Real-Time Multi-person Pose estimation.
> State-of-the-art speed + accuracy balance.
> Used in Pose Estimator (currently commented out — Xovis used instead).
> Better for multi-person scenes than MediaPipe.

**Q61. What is OpenPose?**
> First real-time multi-person pose estimation system.
> Bottom-up approach.
> Uses PAF (Part Affinity Fields) to group keypoints to persons.
> Now largely replaced by RTMPose, ViTPose.

**Q62. What is ViTPose?**
> Vision Transformer based pose estimation.
> Uses ViT backbone → better global context than CNN.
> Current state-of-the-art on COCO keypoint benchmark.

---

# SECTION 9: OBJECT TRACKING

**Q63. What is the difference between detection and tracking?**
```
Detection: Run model on EACH frame independently.
           Fast per frame, no memory of previous frames.
           Same object → different ID each frame.

Tracking:  Follow an object across frames.
           Assign consistent ID.
           Handles temporary occlusion.
           Uses detection + assignment algorithm.
```

**Q64. What is SORT (Simple Online and Realtime Tracking)?**
> Kalman filter + Hungarian algorithm.
> Kalman filter: predicts where object will be next frame.
> Hungarian algorithm: assigns detected boxes to existing tracks.
> Fast, simple, works well when objects rarely occlude.

**Q65. What is DeepSORT?**
> SORT + appearance features (deep learning embeddings).
> Each tracked object has appearance descriptor.
> When object reappears after occlusion → match by appearance, not just position.
> Better identity preservation than SORT.

**Q66. What is ByteTrack?**
> Improved tracker — uses low confidence detections too.
> SORT ignores low confidence boxes → loses objects in occlusion.
> ByteTrack: use low confidence boxes to maintain track continuity.
> State-of-the-art for multi-object tracking.

**Q67. What is IOU tracker?**
> Simplest tracker — match boxes between frames by IoU.
> Box in frame T+1 matched to box in frame T with highest IoU.
> Very fast, works when objects move slowly.
> Fails with fast movement or occlusion.

**Q68. What is Xovis?**
> Commercial people tracking system using stereo cameras.
> Gives: tracking ID, real-world (X,Y) coordinates, person silhouette.
> Uses depth from stereo → can distinguish people even when overlapping in 2D.
> Used in our Pose Estimator as the tracking backend.

---

# SECTION 10: DEPTH ESTIMATION

**Q69. What is depth estimation?**
> Predict distance from camera to each pixel or object.
> Output: depth map (same size as image, each pixel = distance in meters).

**Q70. What are the types of depth estimation?**
```
Stereo depth:
  → Two cameras → disparity → depth
  → Hardware: Intel RealSense, ZED camera, Xovis
  → Accurate, needs calibration

Structured light:
  → Project IR pattern → measure distortion → depth
  → Hardware: Kinect, iPhone Face ID
  → Accurate, limited range

Monocular depth estimation:
  → Single camera → neural network predicts depth
  → No special hardware needed
  → Less accurate, relative depth (not absolute)
  → Models: MiDaS, Depth Anything, ZoeDepth
```

**Q71. What is MiDaS?**
> Monocular depth estimation model by Intel.
> Trained on diverse datasets → generalizes well.
> Output: relative depth (not metric — no absolute distances).
> Used as baseline for monocular depth tasks.

**Q72. What is Depth Anything?**
> Current state-of-the-art monocular depth model.
> Trained on 62 million images — largest depth training set.
> Versions: Depth Anything V1, V2.
> Better generalization than MiDaS.
> Output: relative depth map.

**Q73. What is metric depth vs relative depth?**
```
Relative depth:
  → Pixel A is farther than pixel B
  → No absolute distance (no meters)
  → MiDaS, Depth Anything default output

Metric depth:
  → Pixel A is 2.3 meters away
  → Absolute distance in meters
  → Requires calibration or specialized training
  → ZoeDepth, Depth Anything V2 metric version
```

**Q74. What is a point cloud?**
> 3D representation — collection of (X, Y, Z) points in space.
> Created from: stereo cameras, LiDAR, depth cameras.
> Used in: robotics, autonomous vehicles, 3D mapping.
> Libraries: Open3D, PCL (Point Cloud Library).

---

# SECTION 11: VISION TRANSFORMERS

**Q75. What is ViT (Vision Transformer)?**
> Apply Transformer architecture (from NLP) to images.
> Split image into patches (e.g., 16×16) → treat each patch as a token.
> Self-attention over all patches → global context.
> Limitation: Needs much more data than CNNs to train well.
> On large datasets: better than CNNs.

**Q76. What is the difference between CNN and ViT?**
```
CNN:
  → Local receptive field (kernel sees small area)
  → Inductive bias: translation invariance built-in
  → Works well with less data
  → Faster for small images

ViT:
  → Global attention (every patch attends to every other patch)
  → No inductive bias — learns from data
  → Needs large datasets to match CNN
  → Better at capturing long-range dependencies
```

**Q77. What is Swin Transformer?**
> Hierarchical ViT with local window attention.
> Fixes ViT's problem: attention within windows first, then shift windows.
> Computational complexity linear to image size (ViT is quadratic).
> Better for dense prediction tasks (detection, segmentation).
> Backbone for many SOTA detection/segmentation models.

**Q78. What is DINO/DINOv2?**
> Self-supervised ViT training by Meta.
> Trained without labels — learns features by predicting masked patches.
> DINOv2 features generalize across tasks without fine-tuning.
> Used as backbone for many downstream tasks.

---

# SECTION 12: FOUNDATION MODELS

**Q79. What is CLIP?**
> Contrastive Language-Image Pretraining by OpenAI.
> Trains image encoder + text encoder together.
> Learns: which images match which text descriptions.
> Result: image and text in same embedding space.
> Use cases:
>   - Zero-shot classification (no training on new class)
>   - Image search by text
>   - Image captioning
> Example: "red running shoe" → find matching product images.

**Q80. What is zero-shot classification using CLIP?**
> Classify image into class never seen during training.
> How:
>   1. Convert class names to text embeddings ("a photo of a cat")
>   2. Convert image to image embedding
>   3. Find most similar text embedding → that's the class
> No fine-tuning needed — just change text prompts.

**Q81. What is Grounding DINO?**
> Object detector that accepts text as input.
> "Find the red shoe" → draws bounding box around red shoe.
> Combines DINO (vision) + CLIP (language) + detection head.
> Zero-shot detection — no training for new categories.

**Q82. What is Florence-2?**
> Microsoft's vision foundation model.
> Single model handles: detection, segmentation, captioning, OCR, grounding.
> Prompt-based: change the task by changing the prompt.
> Smaller than GPT-4V but specialized for vision tasks.

**Q83. What is GPT-4V / Claude Vision / Gemini Vision?**
> Large Language Models with vision capability.
> Input: image + text question. Output: text answer.
> Use cases:
>   - Image understanding and reasoning
>   - Visual QA
>   - Document analysis
>   - Product description generation
> Limitation: Slow, expensive, not for real-time CV.

**Q84. What is LLaVA?**
> Open-source Vision Language Model.
> Combine LLaMA + CLIP visual encoder.
> Can answer questions about images.
> Alternative to GPT-4V for local deployment.
> Used in some retail inspection tasks.

---

# SECTION 13: VIDEO UNDERSTANDING

**Q85. What is action recognition?**
> Classify what action is happening in a video.
> Examples: "picking up", "running", "waving"
> Models: SlowFast, VideoMAE, TimeSformer.
> Input: video clip (stack of frames). Output: action label.

**Q86. What is temporal information? Why does it matter?**
> Temporal = across time (across frames).
> Single frame can't tell direction of movement.
> Example: person going up or down stairs looks same in one frame.
> With temporal info: motion direction is clear.
> CNNs miss temporal info → need 3D CNNs or Transformers for video.

**Q87. What is optical flow used for in action recognition?**
> Optical flow = pixel motion vectors between frames.
> Used as additional input alongside RGB frames (two-stream networks).
> Motion information helps distinguish similar-looking actions.

**Q88. What is VideoMAE?**
> Masked Autoencoder for video pretraining.
> Mask 90% of video patches → predict them.
> Forces model to learn temporal structure.
> Current SOTA for video understanding tasks.

---

# SECTION 14: PRODUCTION CV

**Q89. What is model quantization?**
> Reduce model size by using lower precision numbers.
> FP32 → FP16 (half precision) → INT8.
> Result: 2-4x smaller model, faster inference, slight accuracy drop.
> Essential for edge deployment.

**Q90. What is TensorRT?**
> NVIDIA's inference optimization framework.
> Converts PyTorch/ONNX model → optimized engine.
> Techniques: quantization, layer fusion, kernel optimization.
> Result: 2-5x faster inference on NVIDIA GPUs.
> Used in production CV systems.

**Q91. What is ONNX?**
> Open Neural Network Exchange — standard format for ML models.
> Convert PyTorch → ONNX → deploy anywhere (TensorRT, ONNX Runtime, mobile).
> `torch.onnx.export(model, dummy_input, "model.onnx")`

**Q92. What is model pruning?**
> Remove unnecessary weights (near-zero) from model.
> Result: smaller, faster model.
> Types: unstructured (individual weights), structured (whole channels).
> Structured pruning → easier to accelerate on hardware.

**Q93. What is edge computing for CV?**
> Run CV model on device (camera, Raspberry Pi, Jetson) instead of cloud.
> Advantages:
>   - Low latency (no network round trip)
>   - Data privacy (footage stays local)
>   - Works offline
>   - Lower operating cost long-term
> Disadvantages:
>   - Limited compute
>   - Hardware maintenance
>   - Model must be optimized (quantized, pruned)

**Q94. What is Jetson? When is it used?**
> NVIDIA Jetson — edge AI computing module.
> Has GPU → can run deep learning models at edge.
> Jetson Nano → small projects
> Jetson AGX Orin → production, high performance
> Used in autonomous robots, smart cameras, retail CV systems.

**Q95. How do you optimize inference speed for real-time CV?**
> 1. Model selection: use lightweight model (YOLOX-nano vs YOLOX-x)
> 2. Input resolution: smaller image → faster
> 3. Quantization: FP32 → FP16 or INT8
> 4. TensorRT / ONNX Runtime
> 5. Batch processing: process multiple frames together
> 6. ROI processing: run heavy model only on region of interest
> 7. Frame skip: run heavy model every N frames
> 8. Threading: camera read and model inference in parallel threads

**Q96. What is multi-threading in CV systems?**
> Run camera read and model inference in separate threads.
> Without threading: read frame → process → read next → bottleneck.
> With threading:
>   Thread 1 (camera): continuously read frames into buffer
>   Thread 2 (inference): continuously read from buffer and process
> Result: no waiting for camera → max GPU utilization.

**Q97. What is RTSP?**
> Real Time Streaming Protocol — stream video from IP cameras.
> `rtsp://username:password@camera_ip:port/stream`
> Used in Product Handler and Pose Estimator to read camera frames.
> PyAV: Python library to decode RTSP streams efficiently.

**Q98. How do you evaluate a CV model in production?**
> Not just accuracy — also:
>   - Inference latency (ms per frame)
>   - Throughput (FPS)
>   - Memory usage (GPU/RAM)
>   - Model size (for edge deployment)
>   - Power consumption (edge devices)
>   - Accuracy on production data (may differ from test set)
>   - HITL error rate (how often model is uncertain)

---

# SECTION 15: CURRENT STATE OF THE ART (2024-2025)

**Q99. What is the current best object detector?**
> YOLOv9, YOLOv10, YOLOv11 (YOLO family continues to improve)
> RT-DETR: Real-time Detection Transformer (no NMS needed)
> DINO (Detection Transformer with Improved deNoising)
> For production: YOLOv8/v11 (best speed-accuracy balance)

**Q100. What is RT-DETR?**
> Real-Time Detection Transformer by Baidu.
> First real-time transformer-based detector.
> No NMS (Non-Maximum Suppression) needed → end-to-end detection.
> Faster than YOLO at same accuracy level on some benchmarks.

**Q101. What is SAM2 and how does it improve over SAM?**
> SAM2 = Segment Anything Model 2 by Meta.
> SAM1: only images.
> SAM2: images + videos with memory.
> Memory bank: stores object appearance across frames.
> Can segment and track objects through video with single click.

**Q102. What is Depth Anything V2?**
> Improved monocular depth — larger scale training + better synthetic data.
> Metric depth version available (absolute distances).
> Much better than MiDaS on real-world scenes.
> Current best open-source monocular depth model.

**Q103. What are Vision Language Models (VLMs)?**
> Models that understand both images and text.
> Examples: GPT-4V, Claude 3, Gemini, LLaVA, InternVL.
> Use cases:
>   - Describe what's in image
>   - Answer questions about image
>   - Count objects
>   - Detect anomalies with text description
> Trend: replacing specialized CV models for tasks that don't need real-time speed.

**Q104. What is the trend toward foundation models in CV?**
```
Old approach:
  Train separate model for each task:
    → Detector model
    → Segmentation model
    → Classifier model
    → Pose model

New approach (foundation models):
  One large model → handles multiple tasks with prompting
  → SAM: segment anything with prompt
  → CLIP: classify/search anything with text
  → Grounding DINO: detect anything with text
  → Florence-2: detection + segmentation + captioning

Implication: Less task-specific training needed.
             Focus on prompt engineering + fine-tuning.
```

**Q105. What is RAG for CV (Visual RAG)?**
> Retrieval Augmented Generation applied to images.
> Store product images as embeddings (using CLIP).
> Query: "find similar products to this image"
> Retrieve: top-K similar images from vector DB.
> Use in retail: visual product search, recommendation.
> Tools: Qdrant + CLIP embeddings.

---

# SECTION 16: INTERVIEW SCENARIO QUESTIONS

**Q106. You have 95% accuracy but client says system is wrong. What do you do?**
> 1. Check WHICH products are failing (confusion matrix per class)
> 2. Check if test data matches production data (distribution shift)
> 3. Add logging — capture failed predictions with images
> 4. Check confidence threshold — lower confidence = more HITL
> 5. Collect more data for failing classes → retrain

**Q107. Your model is accurate in testing but fails in production. Why?**
> Distribution shift — production data different from training data:
>   - Different lighting in store
>   - Camera angle changed
>   - New product packaging
>   - Seasonal products
> Fix:
>   - Collect production data → add to training set
>   - Test with actual store conditions
>   - Use data augmentation that matches production conditions

**Q108. How do you deploy a CV model for 10 cameras simultaneously?**
> 1. One thread per camera for frame reading
> 2. Shared model on GPU (one model, multiple queues)
> 3. Batch inference — process multiple frames together
> 4. Priority queue — trigger frames get priority
> 5. Monitor GPU utilization — if >90%, scale horizontally (2 servers)

**Q109. Client wants real-time detection but your model runs at 5 FPS. What do you do?**
> Short term:
>   - Reduce input resolution
>   - Use smaller model variant (nano/small)
>   - Quantize to FP16/INT8
>   - TensorRT optimization
>   - Run on ROI only (not full frame)
> Long term:
>   - Upgrade GPU
>   - Use specialized edge hardware (Jetson)
>   - Redesign pipeline with frame skipping

**Q110. How do you handle class imbalance in product classifier? (Some products rarely picked)**
> 1. Weighted loss: rare class gets higher weight
>    `class_weights = {common: 1, rare: 5}`
> 2. Oversample rare class: augmentation to generate more samples
> 3. Focal loss: automatically down-weights easy (common) examples
> 4. Collect more data for rare classes
> 5. Monitor per-class accuracy, not just overall accuracy

---

# QUICK REVISION TABLE

| Topic | Key Model/Tool | Use Case |
|---|---|---|
| Classification | ResNet, EfficientNet, WideResNet | Product type, scene label |
| Detection | YOLOv8, YOLOX, RT-DETR | Person, product, vehicle |
| Segmentation | SAM, Mask RCNN, U-Net | Product region, medical |
| Pose | RTMPose, ViTPose, MediaPipe | Human keypoints |
| Tracking | ByteTrack, DeepSORT, Xovis | Person/vehicle tracking |
| Depth | Depth Anything V2, ZoeDepth | Distance estimation |
| Background sub | MOG2 | Motion detection |
| Feature matching | ORB, SIFT | Image stitching, AR |
| Semantic | Swin+FPN, SegFormer | Road, building segmentation |
| Foundation | CLIP, SAM, Florence-2 | Zero-shot tasks |
| VLM | GPT-4V, LLaVA, InternVL | Image Q&A, captioning |

---

# ANALOGY CHEAT SHEET

| Concept | Analogy |
|---|---|
| Convolution | Sliding magnifying glass scanning image for patterns |
| Pooling | Taking the highlights reel (keep most important) |
| Batch norm | Adjusting everyone's score to same scale before comparing |
| Skip connection | Emergency bypass road when main road (gradient) is blocked |
| Attention | Highlighting which words matter when reading a sentence |
| MOG2 | Security guard who knows what "normal" looks like and flags anything different |
| SAM | Forensic analyst who traces exact outline of what changed |
| Optical flow | Arrows showing which direction each pixel moved |
| Homography | Google Maps calibration — pixel to GPS coordinates |
| Transfer learning | Doctor learning a new specialty — reuses medical knowledge, learns new skills |
| HITL | Quality control team that handles cases the machine isn't sure about |
