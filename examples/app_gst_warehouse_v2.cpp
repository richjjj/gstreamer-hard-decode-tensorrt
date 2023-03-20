#include <signal.h>

#include <thread>

#include "app_gst_warehouse/warehouse_v2.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"
#include "jetson-utils/cudaDraw.h"
#include "jetson-utils/imageIO.h"
#include "jetson-utils/videoSource.h"
using namespace std;

static int test_gstdecode(std::shared_ptr<WarehouseV2::Solution> infer = nullptr) {
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
    uchar3* image = nullptr;
    while (true) {
        auto t0 = iLogger::timestamp_now_float();
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
        if (infer != nullptr) {
            auto t1 = iLogger::timestamp_now_float();
            WarehouseV2::Image infer_image(image, width, height, 0);
            auto result = infer->commit(infer_image);
            auto t2     = iLogger::timestamp_now_float();
            INFO("reuslt=%s", result.c_str());
            INFO("[%d] Capture cost: %.2fms; infer cost: %.2fms.", thread_id, float(t1 - t0), float(t2 - t1));
        }
    }

    // 释放
    SAFE_DELETE(input_stream);
}
int app_warehouse_v2() {
    if (iLogger::exists("hard"))
        iLogger::rmtree("hard");
    iLogger::mkdir("hard");
    auto solution = WarehouseV2::create_solution("yolov5s_pose.FP16.trtmodel");
    // auto yolo     = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::INT8, "yolov6s_qa", 0);
    // warm up
    // for (int i = 0; i < 100; ++i)
    //     solution->commit();

    int num_views = 1;
#pragma omp parallel for num_threads(num_views)
    for (int i = 0; i < num_views; ++i) {
        test_gstdecode(solution);
    }
}