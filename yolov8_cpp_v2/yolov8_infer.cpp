#include "yolov8_infer.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
static Ort::Session* session = nullptr;
static Ort::SessionOptions session_options;

void load_model(const char* model_path) {
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    std::wstring w_model_path(model_path, model_path + strlen(model_path));
    session = new Ort::Session(env, w_model_path.c_str(), session_options);
}

void infer_image(unsigned char* img_bytes, int length, Detection** out_dets, int* out_count) {
    // Decode image
    cv::Mat img = cv::imdecode(cv::Mat(1, length, CV_8UC1, img_bytes), cv::IMREAD_COLOR);
    if (img.empty()) {
        *out_count = 0;
        *out_dets = nullptr;
        return;
    }

    // Resize and convert to float32
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Convert HWC → CHW
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(3 * 640 * 640);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        float* data = (float*)channels[c].data;
        input_tensor_values.insert(input_tensor_values.end(), data, data + 640 * 640);
    }

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    // Get names
    auto input_name_alloc = session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto output_name_alloc = session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    // Run inference
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        &output_name, 1
    );

    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    // Use first detection if confidence > 0.5
    float conf = output_data[4];
    if (conf > 0.5f) {
        int x1 = static_cast<int>(output_data[0]);
        int y1 = static_cast<int>(output_data[1]);
        int x2 = static_cast<int>(output_data[2]);
        int y2 = static_cast<int>(output_data[3]);

        std::vector<Detection> result = {
            {0, conf, {x1, y1, x2, y2}}
        };

        *out_count = static_cast<int>(result.size());
        *out_dets = (Detection*)malloc(sizeof(Detection) * result.size());
        memcpy(*out_dets, result.data(), sizeof(Detection) * result.size());
        return;
    }

    // No valid detection
    *out_count = 0;
    *out_dets = nullptr;
}

void free_detections(Detection* detections) {
    free(detections);
}
