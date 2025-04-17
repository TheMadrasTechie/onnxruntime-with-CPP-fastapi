cmake_minimum_required(VERSION 3.10)
project(YOLOv8CPP)

# ===== Your actual paths =====
set(OpenCV_DIR "C:/opencv/build")
set(OpenCV_BIN_DIR "C:/opencv/build/x64/vc16/bin")
set(OpenCV_DEBUG_DLL_FILENAME "opencv_world4110.dll")
set(OpenCV_RELEASE_DLL_FILENAME "opencv_world4110.dll")
set(ONNXRUNTIME_DIR "C:/onnxruntime/onnxruntime-win-x64-1.21.0")

# ===== Compiler settings =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Include headers if any =====
include_directories(include)

# ===== Only build wrapper.cpp =====
set(SOURCES
    src/wrapper.cpp
)

add_library(yolov8 SHARED ${SOURCES})
target_include_directories(yolov8 PRIVATE "${ONNXRUNTIME_DIR}/include")

# ===== Link OpenCV =====
find_package(OpenCV REQUIRED)
target_link_libraries(yolov8 ${OpenCV_LIBS})

# ===== Link ONNX Runtime =====
target_link_libraries(yolov8 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

# ===== Copy required DLLs after build =====
add_custom_command(TARGET yolov8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    "$<TARGET_FILE_DIR:yolov8>"
)

add_custom_command(TARGET yolov8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_BIN_DIR}/${OpenCV_DEBUG_DLL_FILENAME}"
    "$<TARGET_FILE_DIR:yolov8>"
)

add_custom_command(TARGET yolov8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_BIN_DIR}/${OpenCV_RELEASE_DLL_FILENAME}"
    "$<TARGET_FILE_DIR:yolov8>"
)
