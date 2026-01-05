from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from fastapi.responses import JSONResponse
import torch

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix PyTorch 2.6 weights_only warning
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# Load YOLO model
print("🚀 Loading YOLO model...")
model = YOLO('yolov8n.pt')  # Fastest model
print("✅ YOLO model loaded!")

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'train'
}

frame_count = 0

@app.get("/")
async def root():
    return {"message": "Vehicle Detection Server Running"}

@app.post("/detect-vehicles")
async def detect_vehicles(file: UploadFile = File(...)):
    print("🔥 /detect-vehicles HIT")
    global frame_count
    frame_count += 1

    try:
        print(f"\n{'='*50}")
        print(f"🔍 Processing frame #{frame_count}")

        # Read image bytes
        image_bytes = await file.read()
        print(f"📦 Received bytes: {len(image_bytes)}")

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        print(f"📸 Image shape: {img.shape}")

        # Run YOLO
        results = model(img, conf=0.25, verbose=False)

        vehicles = []
        vehicle_types = {}

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in VEHICLE_CLASSES:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    vehicles.append({
                        "type": class_name,
                        "confidence": round(conf, 2),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })

                    vehicle_types[class_name] = vehicle_types.get(class_name, 0) + 1

        print(f"✅ Vehicles detected: {len(vehicles)}")
        print(f"📋 Vehicle types: {vehicle_types}")
        print(f"{'='*50}")

        return {
            "status": "success",
            "total_vehicles": len(vehicles),
            "vehicle_types": vehicle_types,
            "vehicles": vehicles,
            "frame_number": frame_count
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/test-detection")
async def test_detection():
    """
    Test endpoint - detects objects in a test image
    """
    print("\n🧪 Running test detection...")
    
    # Create a simple test image with some shapes
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_img[:] = (100, 100, 100)  # Gray background
    
    # Draw some rectangles (simulating vehicles)
    cv2.rectangle(test_img, (100, 100), (300, 200), (0, 255, 0), -1)
    cv2.rectangle(test_img, (400, 300), (550, 450), (255, 0, 0), -1)
    
    print(f"📸 Test image shape: {test_img.shape}")
    
    # Run detection
    results = model(test_img, conf=0.25, verbose=True)
    
    detections = []
    for result in results:
        boxes = result.boxes
        print(f"📊 Objects detected in test: {len(boxes)}")
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = float(box.conf[0])
            detections.append({"class": class_name, "conf": conf})
    
    return {
        "test": "completed",
        "detections": detections,
        "model_classes": list(model.names.values())[:20]  # First 20 classes
    }

@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Convert to OpenCV image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")

        # OPTIONAL: Save image for debugging
        cv2.imwrite("received_frame.jpg", img)

        return {"received": True}

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

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