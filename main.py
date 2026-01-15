from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
from torch.nn.modules.container import Sequential
import time
import threading
from pydantic import BaseModel
import os

lock = threading.Lock()

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
model = YOLO('yolov8n.pt')
print("✅ YOLO model loaded!")

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [
    'car', 'motorbike', 'bus', 'truck', 'bicycle', 'train'
]

DETECTION_ENABLED = False
current_vehicle_count = 0
current_traffic_level = 0.0
current_traffic_status = "Unknown"
last_frame_time = None

@app.get("/")
async def root():
    return {"message": "Vehicle Detection Server Running"}

@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    try:
        global current_vehicle_count, current_traffic_level, current_traffic_status, last_frame_time
        
        if not DETECTION_ENABLED:
            return {"detection": "disabled"}

        # Convert to OpenCV image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")
        
        vehicle_count = 0
        results = model(img, stream=True)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in VEHICLE_CLASSES and conf > 0.3:
                    vehicle_count += 1

        with lock:
            current_vehicle_count = vehicle_count
            
            # Calculate traffic level (0-10 scale)
            if vehicle_count == 0:
                current_traffic_level = 0.0
                current_traffic_status = "Clear"
            elif vehicle_count <= 2:
                current_traffic_level = 2.0
                current_traffic_status = "Light"
            elif vehicle_count <= 5:
                current_traffic_level = 5.0
                current_traffic_status = "Moderate"
            elif vehicle_count <= 8:
                current_traffic_level = 7.5
                current_traffic_status = "Heavy"
            else:
                current_traffic_level = 10.0
                current_traffic_status = "Very Heavy"

            last_frame_time = time.time()
            
            print(f"🚗 Vehicles detected: {vehicle_count}")
            print(f"📊 Traffic Level: {current_traffic_level}/10 - {current_traffic_status}")

        return {
            "frame_processed": True,
            "vehicle_count": vehicle_count,
            "traffic_level": current_traffic_level
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

@app.post("/start-detection")
async def start_detection():
    global DETECTION_ENABLED, current_vehicle_count, current_traffic_level, current_traffic_status
    DETECTION_ENABLED = True
    current_vehicle_count = 0
    current_traffic_level = 0.0
    current_traffic_status = "Starting..."
    print("🟢 Detection STARTED")
    return {"status": "started"}

@app.post("/stop-detection")
async def stop_detection():
    global DETECTION_ENABLED
    DETECTION_ENABLED = False
    print("🔴 Detection STOPPED")
    return {"status": "stopped"}

@app.get("/status")
def get_status():
    """Main endpoint - returns traffic status"""
    if not DETECTION_ENABLED:
        return {
            "running": False,
            "traffic_level": 0.0,
            "traffic_status": "Stopped",
            "vehicle_count": 0
        }

    return {
        "running": True,
        "traffic_level": current_traffic_level,
        "traffic_status": current_traffic_status,
        "vehicle_count": current_vehicle_count,
        "last_frame_time": last_frame_time
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🚗 Vehicle Detection Server")
    print("=" * 50)
    print("📡 Endpoints:")
    print("  - POST /frame (receive camera frames)")
    print("  - GET  /status (get traffic status)")
    print("  - POST /start-detection")
    print("  - POST /stop-detection")
    print("=" * 50)
    print("\nDetectable vehicle classes:")
    for vc in VEHICLE_CLASSES:
        print(f"  - {vc}")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))