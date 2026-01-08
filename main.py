from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from sort import Sort
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
#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [
    'car', 'motorbike', 'bus', 'truck', 'bicycle', 'train'
]

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
frame_count = 0

@app.get("/")
async def root():
    return {"message": "Vehicle Detection Server Running"}


@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    try:
        global DETECTION_ENABLED, counted_ids
        # Read image bytes
        

        if not DETECTION_ENABLED:
            return {"detection": "disabled"}
        
        # Convert to OpenCV image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cap = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        success, img = cap.read()

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")
        counted_ids = set()  # define OUTSIDE while loop
        DETECTION_ENABLED = False

        
        results = model(img, stream=True)
        detections = np.empty((0, 5))
        
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in VEHICLE_CLASSES and conf > 0.3:
                    detections = np.vstack(
                        (detections, [x1,y1,x2,y2,conf])
                    )

        tracked = tracker.update(detections)

        frame_count = 0
        for t in tracked:
            _, _, _, _, track_id = t
            track_id = int(track_id)

            if track_id not in counted_ids:
                counted_ids.add(track_id)
                frame_count += 1

        return {
            "frame_processed": True,
            "new_vehicles": frame_count,
            "total_count": len(counted_ids)
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