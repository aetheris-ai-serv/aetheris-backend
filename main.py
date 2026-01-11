from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import cvzone as cvz
import math

app = FastAPI()

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
counted_ids = set()
TOTAL_COUNT = 0
seen_centroids = []
LINE_Y = 10  # counting line position
OFFSET = 2   
DIST_THRESHOLD = 40

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

@app.get("/")
async def root():
    return {"message": "Vehicle Detection Server Running"}


@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    try:
        
        

        if not DETECTION_ENABLED:
            return {"detection": "disabled"}
        
        line_y = 400

        # Convert to OpenCV image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")

        
        results = model(img, stream=True)
        detections = np.empty((0, 5))
        
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
                    if (LINE_Y - OFFSET) < cy < (LINE_Y + OFFSET):

                        is_new = True
                        for (px, py) in seen_centroids:
                            if math.hypot(cx - px, cy - py) < DIST_THRESHOLD:
                                is_new = False
                                break

                        if is_new:
                            TOTAL_COUNT += 1
                            new_vehicles += 1
                            seen_centroids.append((cx, cy))

            

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
    
    uvicorn.run(app, host="0.0.0.0", port=8000)