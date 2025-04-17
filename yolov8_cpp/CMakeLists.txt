cmake_minimum_required(VERSION 3.10)
project(yolov8_cpp_infer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(yolov8 SHARED yolov8_infer.cpp)
target_include_directories(yolov8 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8 ${OpenCV_LIBS} onnxruntime)