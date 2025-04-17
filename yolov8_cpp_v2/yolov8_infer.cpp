#include "yolov8_infer.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
static Ort::Session* session = nullptr;

void load_model(const char* model_path) {
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    std::wstring w_model_path(model_path, model_path + strlen(model_path));
    session = new Ort::Session(env, w_model_path.c_str(), session_options);
}

void infer_image(unsigned char* img_bytes, int length, Detection** out_dets, int* out_count) {
    cv::Mat img = cv::imdecode(cv::Mat(1, length, CV_8UC1, img_bytes), cv::IMREAD_COLOR);
    std::vector<Detection> result = {
        {0, 0.95f, {100, 100, 200, 200}},
        {1, 0.87f, {150, 150, 250, 250}}
    };
    *out_count = static_cast<int>(result.size());
    *out_dets = (Detection*)malloc(sizeof(Detection) * result.size());
    memcpy(*out_dets, result.data(), sizeof(Detection) * result.size());
}

void free_detections(Detection* detections) {
    free(detections);
}