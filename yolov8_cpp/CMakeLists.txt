cmake_minimum_required(VERSION 3.10)
project(yolov8_cpp_infer)

set(CMAKE_CXX_STANDARD 17)

# ---- ONNX Runtime Configuration ----
set(ONNXRUNTIME_INCLUDE_DIR "C:/onnxruntime/onnxruntime-win-x64-1.21.0/include")
set(ONNXRUNTIME_LIB_DIR "C:/onnxruntime/onnxruntime-win-x64-1.21.0/lib")

# ---- OpenCV Configuration ----
set(OpenCV_DIR "C:/opencv/build")  # Adjust only if different

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# ---- Build shared library ----
add_library(yolov8 SHARED yolov8_infer.cpp)
target_link_libraries(yolov8 ${OpenCV_LIBS} onnxruntime)
