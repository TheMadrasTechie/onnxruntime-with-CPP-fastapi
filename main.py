from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from yolov8 import YOLOv8

app = FastAPI()

# Load YOLOv8 model
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read uploaded image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run detection
    boxes, scores, class_ids = yolov8_detector(img)
    annotated_img = yolov8_detector.draw_detections(img)

    # Convert annotated image to JPEG
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
