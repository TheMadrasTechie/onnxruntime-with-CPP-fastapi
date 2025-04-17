#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Detection {
    int class_id;
    float score;
    std::vector<int> box; // x1, y1, x2, y2
};

extern "C" {
    void load_model(const char* model_path);
    void infer_image(unsigned char* img_bytes, int length, Detection** out_dets, int* out_count);
    void free_detections(Detection* detections);
}