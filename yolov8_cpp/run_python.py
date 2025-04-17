from python_wrapper import lib, process_image

# Load the ONNX model (call once)
lib.load_model(b"models/yolov8m.onnx")

# Load and infer image
with open("sample.jpg", "rb") as f:
    image_bytes = f.read()

detections = process_image(image_bytes)
print(detections)
