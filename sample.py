import cv2
import time
from yolov8 import YOLOv8

# Number of simulated feeds
n = 20

# Load YOLOv8 model
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Capture from default webcam
cap = cv2.VideoCapture(0)

# Create window names
for i in range(1, n + 1):
    cv2.namedWindow(f"Feed {i}", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    start_time = time.time()  # Start timing
    # Show same processed frame in all windows
    for i in range(1, n + 1):
        # Run detection once    
        boxes, scores, class_ids = yolov8_detector(frame)
        combined_img = yolov8_detector.draw_detections(frame)
        cv2.imshow(f"Feed {i}", combined_img)
    end_time = time.time()  # End timing
    print(f"Frame processed in {end_time - start_time:.3f} seconds")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
