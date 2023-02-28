#include "warehouse.hpp"

#include <atomic>
#include <chrono>
#include <vector>

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"
#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "jetson-utils/videoSource.h"
#include "track/bytetrack/BYTETracker.h"
namespace Warehouse {
template <typename T>
T round_up(T value, int decimal_places) {
    const T multiplier = std::pow(10.0, decimal_places);
    return std::ceil(value * multiplier) / multiplier;
}
vector<Object> det2tracks(const YoloposeGPUPtr::BoxArray &array) {
    vector<Object> outputs;
    for (int i = 0; i < array.size(); ++i) {
        auto &abox = array[i];
        Object obox;
        obox.prob    = abox.confidence;
        obox.label   = abox.class_label;
        obox.rect[0] = abox.left;
        obox.rect[1] = abox.top;
        obox.rect[2] = abox.right - abox.left;
        obox.rect[3] = abox.bottom - abox.top;
        outputs.emplace_back(obox);
    }
    return outputs;
}
static shared_ptr<YoloGPUPtr::Infer> get_yolo(YoloGPUPtr::Type type, TRT::Mode mode, const string &model,
                                              int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            YoloGPUPtr::image_to_tensor(image, tensor, type, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloGPUPtr::type_name(type),
         mode_name, name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {}, int8process, "inference");
    }

    return YoloGPUPtr::create_infer(model_file,  // engine file
                                    type,        // yolo type, YoloposeGPUPtr::Type::V5 / YoloposeGPUPtr::Type::X
                                    device_id,   // gpu id
                                    0.25f,       // confidence threshold
                                    0.45f,       // nms threshold
                                    YoloGPUPtr::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                    1024                             // max objects
    );
}
static shared_ptr<YoloposeGPUPtr::Infer> get_yolopose(YoloposeGPUPtr::Type type, TRT::Mode mode, const string &model,
                                                      int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            YoloposeGPUPtr::image_to_tensor(image, tensor, type, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloposeGPUPtr::type_name(type),
         mode_name, name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {}, int8process, "inference");
    }

    return YoloposeGPUPtr::create_infer(model_file,  // engine file
                                        type,        // yolo type, YoloposeGPUPtr::Type::V5 / YoloposeGPUPtr::Type::X
                                        device_id,   // gpu id
                                        0.25f,       // confidence threshold
                                        0.45f,       // nms threshold
                                        YoloposeGPUPtr::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                        1024                                 // max objects
    );
}

class SolutionImpl : public Solution {
public:
    virtual ~SolutionImpl() {
        stop();
        INFO("pipeline done.");
    }
    virtual void join() {
        for (auto &t : ts_) {
            if (t.joinable())
                t.join();
        }
    }
    virtual void stop() override {
        for (auto &r : uris_) {
            runnings_[r] = false;
        }
        join();
    }
    virtual vector<string> get_uris() const override {
        return uris_;
    }
    virtual bool make_view(const string &uri, size_t timeout) override {
        promise<bool> pro;
        runnings_[uri] = true;
        ts_.emplace_back(thread(&SolutionImpl::worker, this, uri, ref(pro)));
        bool state = pro.get_future().get();
        if (state) {
            uris_.emplace_back(uri);
        } else {
            INFOE("The uri connection is refused.");
            runnings_[uri] = false;
        }
        return state;
    }
    unique_ptr<BYTETracker> creatTracker() {
        unique_ptr<BYTETracker> tracker(new BYTETracker());
        tracker->config()
            .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_max_time_lost(150);
        return tracker;
    }
    virtual void worker(const string &uri, promise<bool> &state) {
        videoOptions option;
        option.zeroCopy           = false;
        option.codec              = videoOptions::Codec::CODEC_H264;
        option.resource           = uri;
        videoSource *input_stream = videoSource::Create(option);
        if (!input_stream) {
            INFOE("decoder create failed");
            state.set_value(false);
            return;
        }
        state.set_value(true);

        auto tracker = creatTracker();

        if (tracker == nullptr) {
            INFOE("tracker create failed");
            state.set_value(false);
            return;
        }
        void *image = nullptr;
        while (runnings_[uri]) {
            if (!input_stream->Capture(&image, imageFormat::IMAGE_UNKNOWN, 1000)) {
                // check for EOS
                if (!input_stream->IsStreaming())
                    break;
                INFOE("capture failed .");
                continue;
            }
            int width  = input_stream->GetWidth();
            int height = input_stream->GetHeight();
            if (callback_) {
                YoloGPUPtr::Image infer_image((uint8_t *)image, width, height, 0, nullptr,
                                              YoloGPUPtr::ImageType::GPUYUVNV12);
                nlohmann::json tmp_json;
                tmp_json["cameraId"]     = uri;
                tmp_json["det_results"]  = nlohmann::json::array();
                tmp_json["pose_results"] = nlohmann::json::array();
                tmp_json["gcn_results"]  = nlohmann::json::array();
                auto objs_future         = yolo_->commit(infer_image);
                // cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                if (yolo_pose_ != nullptr) {
                    auto objs_pose = yolo_pose_->commit(infer_image).get();
                    auto tracks    = tracker->update(det2tracks(objs_pose));
                    for (size_t t = 0; t < tracks.size(); t++) {
                        auto &obj_pose = objs_pose[tracks[t].detection_index];
                        vector<float> pose(obj_pose.pose, obj_pose.pose + 51);
                        nlohmann::json event_json = {
                            {"id", tracks[t].track_id},
                            {"box", {obj_pose.left, obj_pose.top, obj_pose.right, obj_pose.bottom}},
                            {"pose", pose},
                            {"score", obj_pose.confidence}};
                        tmp_json["pose_results"].emplace_back(event_json);
                        // debug
                        // cv::rectangle(cvimage, cv::Point(obj_pose.left, obj_pose.top),
                        //               cv::Point(obj_pose.right, obj_pose.bottom), cv::Scalar(255, 0, 0), 3);
                        // INFO("box: %s ?= %.2f,%.2f,%.2f,%.2f", event_json["box"].dump().c_str(), obj_pose.left,
                        //      obj_pose.top, obj_pose.right, obj_pose.bottom);
                    }
                }
                auto objs = objs_future.get();
                for (const auto &obj : objs) {
                    nlohmann::json event_json = {{"box", {obj.left, obj.top, obj.right, obj.bottom}},
                                                 {"class_label", 2},  // damo 赋值为2
                                                 {"score", obj.confidence}};
                    tmp_json["det_results"].emplace_back(event_json);
                }
                // debug
                // if (tmp_json["pose_results"].size() > 0) {
                //     cv::putText(cvimage, to_string(frame_index), cv::Point(100, 100), 0, 1, cv::Scalar::all(0),
                //     2,
                //                 16);
                //     cv::imwrite(cv::format("imgs/%03d.jpg", frame_index), cvimage);
                // }
                callback_(2, (void *)&infer_image, (char *)tmp_json.dump().c_str(), tmp_json.dump().size());
            }
        };
        INFO("done %s", uri.c_str());
    }
    virtual void disconnect_view(const string &dis_uri) override {
        runnings_[dis_uri] = false;
        uris_.erase(find(uris_.begin(), uris_.end(), dis_uri));
    }

    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual bool startup(const string &det_name, const string &pose_name, const string &gcn_name, int gpuid,
                         bool use_device_frame) {
        gpu_              = gpuid;
        use_device_frame_ = use_device_frame_;
        if (!pose_name.empty()) {
            yolo_pose_ = get_yolopose(YoloposeGPUPtr::Type::V5, TRT::Mode::FP32, pose_name, gpuid);
            if (yolo_pose_ != nullptr) {
                INFO("yolo_pose will be committed");
                for (int i = 0; i < 10; ++i)
                    yolo_pose_->commit(cv::Mat(640, 640, CV_8UC3)).get();
            }
        }
        auto type = (det_name.find("damo") != string::npos) ? YoloGPUPtr::Type::DAMO : YoloGPUPtr::Type::V5;
        yolo_     = get_yolo(type, TRT::Mode::FP16, det_name, 0);
        if (yolo_ == nullptr) {
            INFOE("create tensorrt engine failed.");
            return false;
        }
        // use_device_frame_ = use_device_frame_;
        // gpu_ = gpu_;
        for (int i = 0; i < 10; ++i)
            yolo_->commit(cv::Mat(640, 640, CV_8UC3)).get();

        return true;
    }

private:
    int gpu_               = 0;
    bool use_device_frame_ = true;
    shared_ptr<YoloposeGPUPtr::Infer> yolo_pose_;
    shared_ptr<YoloGPUPtr::Infer> yolo_;
    vector<thread> ts_;
    vector<string> uris_{};
    map<string, atomic_bool> runnings_;
    ai_callback callback_;
};  // namespace Pipeline
shared_ptr<Solution> create_pipeline(const string &det_name, const string &pose_name, const string &gcn_name, int gpuid,
                                     bool use_device_frame) {
    shared_ptr<SolutionImpl> instance(new SolutionImpl());
    if (!instance->startup(det_name, pose_name, gcn_name, gpuid, use_device_frame)) {
        instance.reset();
    }
    return instance;
}
}  // namespace Warehouse