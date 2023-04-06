#include "app_gst_traffic/traffic.hpp"
#include "common/ilogger.hpp"
#include "infer/trt_infer.hpp"
#include "jetson-utils/cudaDraw.h"
#include "jetson-utils/imageIO.h"
#include "jetson-utils/videoSource.h"

using namespace std;

int app_traffic() {
    auto solution = Traffic::create_solution("yolov6n_v2_reopt_qat_35.0_bnone.INT8.B1.trtmodel");
    std::string raw_data =
        R"({"cameraID":"1","uri":"","events":[{"eventName":"nixing","enable":true,"rois":[{"roiName":"default nixing","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"xingrenchuangru","enable":true,"rois":[{"roiName":"default xingrenchuangru","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"feijidongche","enable":true,"rois":[{"roiName":"default feijidongche","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"biandao","enable":true,"rois":[{"roiName":"变道","pointsNum":2,"points":{"x1":477,"y1":368,"x2":701,"y2":870,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"weiting","enable":true,"rois":[{"roiName":"default weiting","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"yongdu","enable":true,"rois":[{"roiName":"default yongdu","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]}]})";

    int num_views = 2;
#pragma omp parallel for num_threads(num_views)
    for (int i = 0; i < num_views; ++i) {
        auto replaced_string = iLogger::replace_string(raw_data, "1", std::to_string(i), 1);
        auto success         = solution->make_view(replaced_string);
        while (!success) {
            // INFO("failed to make_view[test%d] and retry", i);
            success = solution->make_view(replaced_string);
            iLogger::sleep(5000);
        }
    }

    // 获取摄像头数据
    videoOptions option;
    option.zeroCopy           = false;
    option.codec              = videoOptions::Codec::CODEC_H264;
    option.resource           = "rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101";
    videoSource* input_stream = videoSource::Create(option);
    if (!input_stream) {
        INFO("failed to create input_stream");
        return 1;
    }
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

        auto t1 = iLogger::timestamp_now_float();
        Traffic::Image infer_image(image, width, height, 0, 0);
        auto future = solution->commit(infer_image);
        auto t2     = iLogger::timestamp_now_float();
        auto result = future.get();
        auto t3     = iLogger::timestamp_now_float();
        INFO("[] Capture cost: %.2fms; commit cost: %.2fms; get cost: %.2fms.", float(t1 - t0), float(t2 - t1),
             float(t3 - t2));
        INFO("reuslt=%s", result.c_str());
    }

    // 释放
    SAFE_DELETE(input_stream);
}
