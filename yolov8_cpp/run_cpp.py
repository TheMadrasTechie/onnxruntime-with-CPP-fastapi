import os
import ctypes
import sys

# Step 1: Preload ONNX and OpenCV DLL folders (Windows-safe forward slashes)
try:
    os.add_dll_directory(r"C:/onnxruntime/onnxruntime-win-x64-1.21.0/lib")
    os.add_dll_directory(r"C:/opencv/build/x64/vc16/bin")
except AttributeError:
    print("❌ You need Python 3.8+ to use os.add_dll_directory")
    sys.exit(1)

# Step 2: Define C++ detection structure
class Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("score", ctypes.c_float),
        ("box", ctypes.c_int * 4)
    ]

# Step 3: Load your yolov8.dll
dll_path = r"D:/python/upwork/onnxruntime with CPP fastapi/yolov8_cpp/build/Release/yolov8.dll"
try:
    lib = ctypes.cdll.LoadLibrary(dll_path)
except OSError as e:
    print(f"❌ Failed to load DLL: {dll_path}\n{e}")
    sys.exit(1)

# Step 4: Setup function signatures
lib.load_model.argtypes = [ctypes.c_char_p]
lib.infer_image.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(Detection)), ctypes.POINTER(ctypes.c_int)
]
lib.free_detections.argtypes = [ctypes.POINTER(Detection)]

# Step 5: Wrapper for image inference
def process_image(img_bytes: bytes):
    img_ptr = (ctypes.c_ubyte * len(img_bytes))(*img_bytes)
    dets_ptr = ctypes.POINTER(Detection)()
    count = ctypes.c_int()

    lib.infer_image(img_ptr, len(img_bytes), ctypes.byref(dets_ptr), ctypes.byref(count))

    results = []
    for i in range(count.value):
        d = dets_ptr[i]
        results.append({
            "class_id": d.class_id,
            "score": round(d.score, 4),
            "box": list(d.box)
        })

    lib.free_detections(dets_ptr)
    return results

# ========== RUN HERE ==========

# Step 6: Load the ONNX model
model_path = b"models/yolov8m.onnx"
print(f"📦 Loading model: {model_path.decode()}")
lib.load_model(model_path)

# Step 7: Load test image
image_path = "sample.jpg"
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

with open(image_path, "rb") as f:
    image_bytes = f.read()

# Step 8: Run detection
print("🚀 Running detection...")
detections = process_image(image_bytes)
print("✅ Detections:", detections)
