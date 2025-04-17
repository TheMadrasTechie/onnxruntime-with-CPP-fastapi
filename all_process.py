import cv2
import time
from yolov8 import YOLOv8

# Load YOLOv8 model
model_path = "models/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture from webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv8 Feed", cv2.WINDOW_NORMAL)

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    boxes, scores, class_ids = yolov8_detector(frame)
    result_img = yolov8_detector.draw_detections(frame)

    # Face detection and draw rectangles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # FPS
    fps_val = 1.0 / (time.time() - start_time)
    cv2.putText(result_img, f"FPS: {fps_val:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show feed
    cv2.imshow("YOLOv8 Feed", result_img)

    # Save video
    out.write(result_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()