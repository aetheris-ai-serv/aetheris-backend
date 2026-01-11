from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import math
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
from torch.nn.modules.container import Sequential
import time
from pydantic import BaseModel
import os

torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    Sequential
])

app = FastAPI()

# ===================== DATA MODEL =====================
class SpeedData(BaseModel):
    speed_kmph: float
    timestamp: float | None = None

# ===================== STORAGE (IN-MEMORY) =====================
latest_speed = {
    "speed_kmph": 0.0,
    "timestamp": None
}

@app.post("/speed")
def receive_speed(data: SpeedData):
    latest_speed["speed_kmph"] = data.speed_kmph
    latest_speed["timestamp"] = data.timestamp or time.time()

    print(f"🚗 Speed received: {data.speed_kmph:.2f} km/h")

    return {
        "status": "ok",
        "speed_kmph": data.speed_kmph
    }

# ------------------------------------------------
@app.get("/speed")
def get_speed():
    return latest_speed


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load YOLO model
print("🚀 Loading YOLO model...")
model = YOLO('yolov8n.pt')  # Fastest model
print("✅ YOLO model loaded!")

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [
    'car', 'motorbike', 'bus', 'truck', 'bicycle', 'train'
]
DETECTION_ENABLED = False
TOTAL_COUNT = 0
seen_centroids = []
LINE_X = 10  # counting line position 
DIST_THRESHOLD = 40


@app.get("/")
async def root():
    return {"message": "Vehicle Detection Server Running"}


@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    try:
        global TOTAL_COUNT, seen_centroids
        new_vehicles = 0
        if not DETECTION_ENABLED:
            return {"detection": "disabled"}
        

        # Convert to OpenCV image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")

        
        results = model(img, stream=True)
        
        
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in VEHICLE_CLASSES and conf > 0.3:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Check if centroid is near counting line
                    if (LINE_X ) < cx < (int(img.shape[1])) :

                        is_new = True
                        for (px, py) in seen_centroids:
                            if math.hypot(cx - px, cy - py) < DIST_THRESHOLD:
                                is_new = False
                                break

                        if is_new:
                            TOTAL_COUNT += 1
                            new_vehicles += 1
                            seen_centroids.append((cx, cy))
                            if len(seen_centroids) > 20:
                                seen_centroids.pop(0)
        return {
            "frame_processed": True,
            "new_vehicles": new_vehicles,
            "total_count": TOTAL_COUNT
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

@app.post("/start-detection")
async def start_detection():
    global DETECTION_ENABLED, counted_ids
    DETECTION_ENABLED = True
    counted_ids.clear()
    print("🟢 Detection STARTED")
    return {"status": "started"}

@app.post("/stop-detection")
async def stop_detection():
    global DETECTION_ENABLED
    DETECTION_ENABLED = False
    print("🔴 Detection STOPPED")
    return {"status": "stopped"}

@app.post("/debug")
async def debug(data: dict):
    print("🔥 DEBUG HIT")
    print(data.keys())
    return {"received": True}

if __name__ == "__main__":
    import uvicorn
    print("="*50)
    print("🚗 Vehicle Detection Server (DEBUG MODE)")
    print("="*50)
    print("📡 Endpoint: http://localhost:8000/detect-vehicles")
    print("🧪 Test endpoint: http://localhost:8000/test-detection")
    print("="*50)
    print("\nDetectable vehicle classes:")
    for vc in VEHICLE_CLASSES:
        print(f"  - {vc}")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))