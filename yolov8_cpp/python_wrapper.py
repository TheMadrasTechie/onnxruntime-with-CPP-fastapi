import os
import ctypes

# ✅ Add runtime DLL paths for ONNX Runtime and OpenCV
os.add_dll_directory(r"C:\onnxruntime\onnxruntime-win-x64-1.21.0\lib")
os.add_dll_directory(r"C:\opencv\build\x64\vc16\bin")

class Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("score", ctypes.c_float),
        ("box", ctypes.c_int * 4)
    ]

# ✅ Load your compiled C++ inference DLL
lib = ctypes.cdll.LoadLibrary(r"D:\python\upwork\onnxruntime with CPP fastapi\yolov8_cpp\build\Release\yolov8.dll")

# ✅ Setup C function signatures
lib.load_model.argtypes = [ctypes.c_char_p]
lib.infer_image.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(Detection)), ctypes.POINTER(ctypes.c_int)
]
lib.free_detections.argtypes = [ctypes.POINTER(Detection)]

# ✅ Python wrapper for C++ inference
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
