#include <signal.h>

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "builder/trt_builder.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"
#include "jetson-utils/cudaDraw.h"
#include "jetson-utils/imageIO.h"
#include "jetson-utils/videoSource.h"
using namespace std;
static const char* cocolabels[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};
static shared_ptr<YoloGPUPtr::Infer> get_yolo(YoloGPUPtr::Type type, TRT::Mode mode, const string& model,
                                              int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            YoloGPUPtr::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloGPUPtr::type_name(type),
         mode_name, name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    int test_batch_size = 16;
    string model_file   = iLogger::format("%s.%s.B%d.trtmodel", name, mode_name, test_batch_size);

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {}, int8process, "inference");
    }

    return YoloGPUPtr::create_infer(model_file,  // engine file
                                    type,        // yolo type, YoloGPUPtr::Type::V5 / YoloGPUPtr::Type::X
                                    device_id,   // gpu id
                                    0.25f,       // confidence threshold
                                    0.45f,       // nms threshold
                                    YoloGPUPtr::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                    1024                             // max objects
    );
}
bool signal_recieved = false;

static void sig_handler(int signo) {
    if (signo == SIGINT) {
        INFO("received SIGINT");
        signal_recieved = true;
    }
}

static int test_gstdecode(std::shared_ptr<YoloGPUPtr::Infer> infer = nullptr) {
    auto thread_id = std::this_thread::get_id();
    int frame_count{0};
    // create input stream
    videoOptions option;
    option.zeroCopy = false;
    option.codec    = videoOptions::Codec::CODEC_H264;
    option.resource = "rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101";
    // option.resource = "rtsp://admin:dahua123456@39.184.150.8:10554/cam/realmonitor?channel=2&subtype=0";
    videoSource* input_stream = videoSource::Create(option);
    if (!input_stream) {
        INFO("failed to create input_stream");
        return 1;
    }

    // process loop
    // uchar3* image = nullptr;
    void* image = nullptr;
    while (!signal_recieved) {
        auto t0 = iLogger::timestamp_now_float();
        if (!input_stream->Capture(&image, imageFormat::IMAGE_UNKNOWN, 1000)) {
            // check for EOS
            if (!input_stream->IsStreaming())
                break;
            INFOE("capture failed .");
            continue;
        }
        int width  = input_stream->GetWidth();
        int height = input_stream->GetHeight();

        // infer
        if (infer != nullptr) {
            auto t1 = iLogger::timestamp_now_float();
            YoloGPUPtr::Image infer_image((uint8_t*)image, width, height, 0, nullptr,
                                          YoloGPUPtr::ImageType::GPUYUVNV12);
            auto result = infer->commit(infer_image).get();
            auto t2     = iLogger::timestamp_now_float();
            INFO("[%d] Capture cost: %.2fms; infer cost: %.2fms.", thread_id, float(t1 - t0), float(t2 - t1));
        }
    }

    // 释放
    SAFE_DELETE(input_stream);
}
int app_gstdecode() {
    if (iLogger::exists("hard"))
        iLogger::rmtree("hard");
    iLogger::mkdir("hard");
    auto yolo = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::INT8, "yolov6s_qa", 0);
    // warm up
    for (int i = 0; i < 100; ++i)
        yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();

    int num_views = 1;
#pragma omp parallel for num_threads(num_views)
    for (int i = 0; i < num_views; ++i) {
        test_gstdecode(yolo);
    }
}