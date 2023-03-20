#include "warehouse_v2.hpp"

#include <sstream>

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "track/bytetrack/BYTETracker.h"
namespace WarehouseV2 {
using namespace std;
vector<Object> det2tracks(const YoloGPUPtr::BoxArray& array) {
    vector<Object> outputs;
    for (int i = 0; i < array.size(); ++i) {
        auto& abox = array[i];
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

vector<Object> det2tracks(const YoloposeGPUPtr::BoxArray& array) {
    vector<Object> outputs;
    for (int i = 0; i < array.size(); ++i) {
        auto& abox = array[i];
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
string Box2string(const Box& b) {
    stringstream ss;
    ss << "box=[" << b.left << " " << b.top << " " << b.right << " " << b.bottom << "],"
       << "confidence=" << b.confidence << ",class_lable=" << b.class_label << ",id=" << b.id;
    return ss.str();
}
string BoxArray2string(const BoxArray& ba) {
    stringstream ss;
    ss << "[";
    for (auto& b : ba) {
        ss << Box2string(b) << ";";
    }
    ss << "]";
    return ss.str();
}
class SolutionImpl : public Solution {
public:
    virtual bool startup(const string& det_name) {
        // yolo_      = YoloGPUPtr::create_infer(det_name, YoloGPUPtr::Type::V5, 0);
        yolo_pose_ = YoloposeGPUPtr::create_infer(det_name, YoloposeGPUPtr::Type::V5, 0);
        // warm up

        tracker_ = make_shared<BYTETracker>();
        tracker_->config()
            .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_max_time_lost(150);
        if (yolo_pose_ == nullptr || tracker_ == nullptr) {
            return false;
        }
        return true;
    }
    virtual std::string commit(const Image& image) override {
        nlohmann::json tmp_json;
        tmp_json["cameraId"]     = image.camera_id;
        tmp_json["det_results"]  = nlohmann::json::array();
        tmp_json["pose_results"] = nlohmann::json::array();
        tmp_json["gcn_results"]  = nlohmann::json::array();

        // 初始化
        YoloGPUPtr::Image infer_image;

        if (image.device_id < 0) {
            cv::Mat tmp(image.height, image.width, CV_8UC3, (uint8_t*)image.bgrptr);
            infer_image = tmp;
        } else {
            infer_image = YoloGPUPtr::Image((uint8_t*)image.bgrptr, image.width, image.height, image.device_id, nullptr,
                                            YoloGPUPtr::ImageType::GPUBGR);
        }
        if (yolo_pose_ != nullptr) {
            auto t1        = iLogger::timestamp_now_float();
            auto objs_pose = yolo_pose_->commit(infer_image).get();
            auto t2        = iLogger::timestamp_now_float();
            auto tracks    = tracker_->update(det2tracks(objs_pose));
            auto t3        = iLogger::timestamp_now_float();

            INFO("Infer cost: %.2fms; track cost: %.2fms.", float(t2 - t1), float(t3 - t2));
            for (size_t t = 0; t < tracks.size(); t++) {
                auto& obj_pose = objs_pose[tracks[t].detection_index];
                vector<float> pose(obj_pose.pose, obj_pose.pose + 51);
                nlohmann::json event_json = {{"id", tracks[t].track_id},
                                             {"box", {obj_pose.left, obj_pose.top, obj_pose.right, obj_pose.bottom}},
                                             {"pose", pose},
                                             {"score", obj_pose.confidence}};
                tmp_json["pose_results"].emplace_back(event_json);
            }
        }
        return tmp_json.dump();
    }
    virtual vector<string> commits(const std::vector<Image>& images) override {
        vector<string> rs;
        for (auto& image : images) {
            rs.emplace_back(commit(image));
        }
        return rs;
    }

private:
    bool use_device_frame_ = true;
    shared_ptr<YoloposeGPUPtr::Infer> yolo_pose_;
    shared_ptr<YoloGPUPtr::Infer> yolo_;
    shared_ptr<BYTETracker> tracker_;
};
std::shared_ptr<Solution> create_solution(const std::string& det_name) {
    shared_ptr<SolutionImpl> instance(new SolutionImpl());
    if (!instance->startup(det_name)) {
        instance.reset();
    }
    return instance;
}
}  // namespace WarehouseV2