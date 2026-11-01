import cv2, redis, torch, numpy as np, threading, queue, time, gc
from torchvision import models, transforms
from PIL import Image

# config
RTSP_URL  = "rtsp://camera_ip/stream"
CONFIDENCE = 0.7

# redis
iot_db     = redis.Redis(db=0)
product_db = redis.Redis(db=2)

# shared state
frame_queue   = queue.Queue(maxsize=80)
trigger_event = threading.Event()

# ── 1. CAMERA THREAD ──────────────────────────────────────
def camera_thread():
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(2); cap = cv2.VideoCapture(RTSP_URL); continue
        if frame_queue.full(): frame_queue.get()
        frame_queue.put(frame)

# ── 2. TRIGGER MONITOR THREAD ─────────────────────────────
def trigger_monitor():
    while True:
        if iot_db.lpop("triggers"):
            trigger_event.set()
        time.sleep(0.1)

# ── 3. WORKER THREAD ──────────────────────────────────────
def worker_thread():
    while True:
        trigger_event.wait()
        trigger_event.clear()
        frames = list(frame_queue.queue)
        if len(frames) >= 10:
            process(frames)

# ── 4. MOG2 — get before/after frames ─────────────────────
def get_action_window(frames):
    mog2 = cv2.createBackgroundSubtractorMOG2(varThreshold=40, detectShadows=False)
    scores = [np.sum(mog2.apply(f) > 0) for f in frames]
    thresh = np.mean(scores) * 0.3
    before = max((i for i, s in enumerate(scores) if s < thresh), default=0)
    after  = next((i for i in range(before, len(scores)) if scores[i] < thresh), -1)
    return frames[before], frames[after]

# ── 5. DIFF — get changed region ──────────────────────────
def get_changed_region(before, after):
    diff     = cv2.absdiff(before, after)
    gray     = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    cleaned  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return after[y:y+h, x:x+w]

# ── 6. RESNET CLASSIFIER ──────────────────────────────────
SKUS = ["Basmati Rice", "Brown Rice", "Tata Salt"]  # 300 products
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(SKUS))
model.load_state_dict(torch.load("product_classifier.pth"))
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out  = model(tfm(img).unsqueeze(0))
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    return SKUS[pred.item()], conf.item()

# ── MAIN PIPELINE ─────────────────────────────────────────
def process(frames):
    before, after = get_action_window(frames)
    crop = get_changed_region(before, after)
    if crop is None: return

    sku, conf = classify(crop)
    if conf < CONFIDENCE: return

    product_db.hset("latest_detection", mapping={"product": sku, "confidence": round(conf, 2)})
    print(f"Detected: {sku} ({conf:.2f})")

    del before, after, crop, frames
    gc.collect()

# ── STARTUP ───────────────────────────────────────────────
if __name__ == "__main__":
    for target in [camera_thread, trigger_monitor, worker_thread]:
        threading.Thread(target=target, daemon=True).start()
    print("Product Handler running...")
    threading.Event().wait()  # keep main alive
