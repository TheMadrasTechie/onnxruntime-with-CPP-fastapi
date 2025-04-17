import cv2
import numpy as np
import time
from yolov8 import YOLOv8

# Load YOLOv8 model once
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

def process_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    start_time = time.time()
    boxes, scores, class_ids = yolov8_detector(img)
    end_time = time.time()

    result = []
    for box, score, cls_id in zip(boxes, scores, class_ids):
        result.append({
            "class_id": int(cls_id),
            "score": float(score),
            "box": [int(i) for i in box]
        })

    return {
        "time_taken": round(end_time - start_time, 4),
        "detections": result        
    }
