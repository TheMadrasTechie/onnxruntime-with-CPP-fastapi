#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8");
static std::unique_ptr<Ort::Session> session;
static Ort::SessionOptions session_options;

extern "C" __declspec(dllexport)
void load_model(const char* model_path) {
    std::wstring wpath(model_path, model_path + strlen(model_path));
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    session = std::make_unique<Ort::Session>(env, wpath.c_str(), session_options);
}

extern "C" __declspec(dllexport)
int run_inference(const char* image_path, float* out_boxes, int max_boxes) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return 0;

    cv::resize(img, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values;
    std::vector<cv::Mat> chw(3);
    cv::split(img, chw);
    for (int c = 0; c < 3; ++c)
        input_tensor_values.insert(input_tensor_values.end(), (float*)chw[c].data, (float*)chw[c].data + 640 * 640);

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;

auto input_name_ptr = session->GetInputName(0, allocator);
auto output_name_ptr = session->GetOutputName(0, allocator);

auto output_tensors = session->Run(Ort::RunOptions{nullptr},
    &input_name_ptr, &input_tensor, 1,
    &output_name_ptr, 1
);

allocator.Free(input_name_ptr);
allocator.Free(output_name_ptr);


    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    if (output_data[4] > 0.5f) {
        out_boxes[0] = 0;                      // class_id
        out_boxes[1] = output_data[4];         // confidence
        out_boxes[2] = output_data[0];         // x
        out_boxes[3] = output_data[1];         // y
        out_boxes[4] = output_data[2];         // w
        out_boxes[5] = output_data[3];         // h
        return 1;
    }

    return 0;
}
