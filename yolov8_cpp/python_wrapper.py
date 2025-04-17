import ctypes

class Detection(ctypes.Structure):
    _fields_ = [("class_id", ctypes.c_int),
                ("score", ctypes.c_float),
                ("box", ctypes.c_int * 4)]

lib = ctypes.cdll.LoadLibrary("./build/libyolov8.so")

lib.load_model.argtypes = [ctypes.c_char_p]
lib.infer_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                            ctypes.POINTER(ctypes.POINTER(Detection)), ctypes.POINTER(ctypes.c_int)]
lib.free_detections.argtypes = [ctypes.POINTER(Detection)]

def process_image(img_bytes):
    ptr = (ctypes.c_ubyte * len(img_bytes))(*img_bytes)
    dets_ptr = ctypes.POINTER(Detection)()
    count = ctypes.c_int()

    lib.infer_image(ptr, len(img_bytes), ctypes.byref(dets_ptr), ctypes.byref(count))

    results = []
    for i in range(count.value):
        det = dets_ptr[i]
        results.append({
            "class_id": det.class_id,
            "score": float(det.score),
            "box": list(det.box)
        })

    lib.free_detections(dets_ptr)
    return results