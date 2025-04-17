cmake_minimum_required(VERSION 3.10)
project(YOLOv8ONNXRuntime)

# Set ONNX and OpenCV paths
set(OpenCV_DIR "C:/opencv/build")
set(OpenCV_BIN_DIR "C:/opencv/build/x64/vc16/bin")
set(OpenCV_DLL "opencv_world4110.dll")  # change if you use a different version

set(ONNXRUNTIME_DIR "C:/onnxruntime/onnxruntime-win-x64-1.21.0")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${OpenCV_DIR}/include)

# Add your main source
add_executable(yolov8_app main.cpp)

# Find and link OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(yolov8_app ${OpenCV_LIBS})

# Link ONNX Runtime
target_link_libraries(yolov8_app "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

# Copy DLLs after build
add_custom_command(TARGET yolov8_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    "$<TARGET_FILE_DIR:yolov8_app>"
)

add_custom_command(TARGET yolov8_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_BIN_DIR}/${OpenCV_DLL}"
    "$<TARGET_FILE_DIR:yolov8_app>"
)
