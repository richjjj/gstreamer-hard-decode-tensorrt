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
    int test_batch_size = 8;
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
int app_gstdecode() {
    if (iLogger::exists("hard"))
        iLogger::rmtree("hard");
    iLogger::mkdir("hard");
    auto yolo = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::FP16, "yolov6n", 0);
    // warm up
    for (int i = 0; i < 100; ++i)
        yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();

    int frame_count{0};
    // create input stream
    videoOptions option;
    option.zeroCopy = false;
    option.codec    = videoOptions::Codec::CODEC_H264;
    // option.resource = "rtsp://admin:dahua123456@39.184.150.8:10554/cam/realmonitor?channel=2&subtype=0";
    // option.codec    = videoOptions::Codec::CODEC_H264;
    option.resource = "rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101";
    option.Print();
    videoSource* input_stream = videoSource::Create(option);
    if (!input_stream) {
        INFO("failed to create input_stream");
        return 1;
    }

    // process loop
    while (!signal_recieved) {
        uchar3* image = nullptr;
        // void* image;
        if (!input_stream->Capture(&image, 1000)) {
            // check for EOS
            if (!input_stream->IsStreaming())
                break;
            INFOE("capture failed .");
            continue;
        }
        int width  = input_stream->GetWidth();
        int height = input_stream->GetHeight();

        // infer
        auto t1 = iLogger::timestamp_now_float();
        YoloGPUPtr::Image infer_image((uint8_t*)image, width, height, 0, nullptr, YoloGPUPtr::ImageType::GPUBGR);
        auto result = yolo->commit(infer_image).get();
        auto t2     = iLogger::timestamp_now_float();
        INFO("infer cost %.2fms.", float(t2 - t1));

        // draw
        // cv::Mat cvimage(height, width, CV_8UC3);
        // cudaMemcpy(cvimage.data, image, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost);
        // for (auto& obj : result) {
        //     auto name      = cocolabels[obj.class_label];
        //     auto caption   = iLogger::format("%s %.2f", name, obj.confidence);
        //     int font_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        //     cv::rectangle(cvimage, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
        //                   cv::Scalar(0, 0, 255), 5);
        //     cv::rectangle(cvimage, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + font_width, obj.top),
        //                   cv::Scalar(255, 0, 0), -1);
        //     cv::putText(cvimage, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        // }
        // ++frame_count;
        // auto img_path = iLogger::format("hard/%d.jpg", frame_count);
        // cv::imwrite(img_path, cvimage);
        // cuda draw
        // for (auto& obj : result) {
        //     CUDA(cudaDrawRect(image, width, height, obj.left, obj.top, obj.right, obj.bottom,
        //                       make_float4(255, 0, 0, 200)));  // color:x y z w ;w>0 表示填充
        // }
        // ++frame_count;
        // auto img_path = iLogger::format("hard/%d.jpg", frame_count);

        // saveImage(img_path.c_str(), image, width, height);
    }

    // 释放
    SAFE_DELETE(input_stream);
    return 0;
}