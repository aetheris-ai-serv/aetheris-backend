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
    'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'train'
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

# @app.post("/detect-vehicles")
# async def detect_vehicles(file: UploadFile = File(...)):
    

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
        cap = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        success, img = cap.read()

        if img is None:
            return {"error": "Invalid image"}

        print(f"📸 Frame received: {img.shape}")
        counted_ids = set()  # define OUTSIDE while loop

        while True:
            results = model(img, stream=True)
            detections = np.empty((0, 5))
        
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)      

                    w,h = x2-x1, y2-y1
                    cvz.cornerRect(img,(x1,y1,w,h))
                    conf = math.ceil((box.conf[0]*100))/100
                    
                    #class names
                    cls =int(box.cls[0])
                    currentClass = classNames[cls]

                    if currentClass in VEHICLE_CLASSES and conf > 0.3:
                        cvz.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale =0.6,thickness=1,offset=2)
                        cvz.cornerRect(img,(x1,y1,w,h),l=9)
                        currentArray = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack((detections,currentArray))

            resultsTracker = tracker.update(detections)
            for result in resultsTracker:
                x1,y1,x2,y2,id = result
                x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
                cvz.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=2,colorR=(255,0,0))
                cvz.putTextRect(img,f'ID: {id}',(x1, y1-10),scale=1.5)

            if id not in counted_ids:
                counted_ids.add(id)
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