import random
from fastapi import FastAPI,File, UploadFile
import cv2
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import uvicorn
import GDRegressor 

app = FastAPI()

origins = [
    "*",]


app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],)

last_frame_time = 0
FRAME_INTERVAL = 0.3  # seconds (≈ 3 FPS)

class SpeedData(BaseModel):
    speed: float

@app.post("/frame")
async def receive_frame(frame: UploadFile = File(...)):
    print("📸 Frame received")
    global last_frame_time

    current_time = time.time()
    if current_time - last_frame_time < FRAME_INTERVAL:
        return {"status": "skipped"}  # throttle protection

    last_frame_time = current_time

    # Read image bytes
    image_bytes = await frame.read()

    # Convert bytes → NumPy
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode JPEG → OpenCV image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}
    else:
        print("Frame received")
    height, width, _ = image.shape

    # Example: draw timestamp
    cv2.putText(
        image,
        time.strftime("%H:%M:%S"),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Example: convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Example: edge detection
    edges = cv2.Canny(gray, 100, 200)

    # -------------------------------
    # OPTIONAL: Display frame (DEV ONLY)
    # -------------------------------
    cv2.imshow("Camera Feed", image)
    cv2.imshow("Edges", edges)
    cv2.waitKey(1)

    return {
        "status": "frame received",
        "width": width,
        "height": height,
    }

# -------------------------------
# SHUTDOWN CLEANUP
# -------------------------------
@app.on_event("shutdown")
def shutdown_event():
    cv2.destroyAllWindows()


@app.get("/")
def root():
    return {"status": "FastAPI is running"}

@app.post("/speed")
def receive_speed(data: SpeedData):
    print(f"Received speed: {data.speed} km/h")
    xspeed = data.speed
    
    return {
        "received_speed": data.speed,
        "message": "Speed received successfully"
    }




if __name__ == "__main__":
    # HOST: 0.0.0.0 allows access from external devices (your phone)
    uvicorn.run(app, host="0.0.0.0", port=8000)
