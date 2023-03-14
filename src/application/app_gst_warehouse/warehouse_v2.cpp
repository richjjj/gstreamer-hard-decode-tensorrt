#include "warehouse_v2.hpp"

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"
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
class SolutionImpl : public Solution {
public:
    virtual bool startup(const string& det_name) {
        yolo_ = YoloGPUPtr::create_infer(det_name, YoloGPUPtr::Type::V5, 0);
        // warm up

        tracker_ = make_shared<BYTETracker>();
        tracker_->config()
            .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_max_time_lost(150);
        if (yolo_ == nullptr || tracker_ == nullptr) {
            return false;
        }
        return true;
    }
    virtual std::string commit(const Image& image) override {
        YoloGPUPtr::Image infer_image((uint8_t*)image.bgrptr, image.width, image.height, image.device_id, nullptr,
                                      YoloGPUPtr::ImageType::GPURGB);

        auto objs   = yolo_->commit(infer_image).get();
        auto tracks = tracker_->update(det2tracks(objs));
        BoxArray output;
        for (size_t t = 0; t < tracks.size(); t++) {
            auto& track = tracks[t];
            auto obj    = objs[track.detection_index];
            output.emplace_back(obj.left, obj.top, obj.right, obj.bottom, obj.confidence, obj.class_label,
                                track.track_id);
        }
        return "";
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