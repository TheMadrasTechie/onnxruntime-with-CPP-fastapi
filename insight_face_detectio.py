import cv2
import time
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Open webcam
cap = cv2.VideoCapture(0)

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("insightface_output.avi", fourcc, fps, (width, height))

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        gender = "Male" if face.gender == 1 else "Female"
        age = int(face.age)

        # Draw on frame
        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"{gender}, {age}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"Detected {gender}, Age {age}")

    # FPS
    fps_val = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps_val:.2f}")

    # Write frame to video
    out.write(frame)

    # Optional: Stop on keyboard interrupt (Ctrl+C)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
