#include "traffic.hpp"

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "event.hpp"

namespace Traffic {
using namespace std;
class SolutionImpl : public Solution {
public:
    virtual bool startup(const string& det_name) {
        yolo_ = YoloGPUPtr::create_infer(det_name, YoloGPUPtr::Type::V5, 0);
        if (yolo_ == nullptr) {
            return false;
        }
        for (int i = 0; i < 20; ++i) {
            // warm up
            yolo_->commit(cv::Mat(640, 640, CV_8UC3)).get();
        }
        return true;
    }
    virtual bool make_view(const string& raw_data, size_t timeout) override {
        // 创建eventinfer对象 和camera_id对应
        auto event_infer = create_event(raw_data);
        if (!event_infer) {
            INFOE("The uri connection is refused.");
            event_infer.reset();
            return false;
        }
        int camera_id            = event_infer->get_cameraID();
        event_infers_[camera_id] = std::move(event_infer);
        // event_infers_.insert(make_pair(camera_id, event_infer));
        INFO("make_view done: {cameraID[%d]:event_infer}", camera_id);
        return true;
    }
    virtual std::shared_future<std::string> commit(const Image& image) override {
        // for (auto ev : event_infers_) {
        //     INFO("%d", ev.first);
        // }
        auto event_infer = event_infers_.find(image.camera_id);
        if (event_infer == event_infers_.end()) {
            INFOW("please  make_view cmaraID[%d]", image.camera_id);
        }
        // 初始化
        YoloGPUPtr::Image infer_image;

        if (image.device_id < 0) {
            cv::Mat tmp(image.height, image.width, CV_8UC3, (uint8_t*)image.bgrptr);
            infer_image = tmp;
        } else {
            infer_image = YoloGPUPtr::Image((uint8_t*)image.bgrptr, image.width, image.height, image.device_id, nullptr,
                                            YoloGPUPtr::ImageType::GPUBGR);
        }
        return event_infer->second->commit(yolo_->commit(infer_image));
    }
    virtual std::vector<std::shared_future<std::string>> commits(const std::vector<Image>& images) override {
        ;
    }

private:
    shared_ptr<YoloGPUPtr::Infer> yolo_;
    vector<thread> ts_;
    vector<string> uris_;
    map<int, shared_ptr<EventInfer>> event_infers_;
    vector<int> cameraIDs_;
    map<string, atomic_bool> runnings_;
};

std::shared_ptr<Solution> create_solution(const std::string& det_name) {
    shared_ptr<SolutionImpl> instance(new SolutionImpl());
    if (!instance->startup(det_name)) {
        instance.reset();
    }
    return instance;
};
}  // namespace Traffic