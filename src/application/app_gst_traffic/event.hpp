#ifndef __EVENT_HPP__
#define __EVENT_HPP__
#include <memory>
#include <string>

#include "app_yolo_gpuptr/yolo_gpuptr.hpp"

namespace Traffic {

struct RoiConfig {
    std::string roiName;
    int pointsNum;
    std::vector<cv::Point2f> points;
};
struct EventConfig {
    std::string eventName;
    bool enable{false};
    std::vector<RoiConfig> rois;
};
struct ViewConfig {
    std::string cameraID;
    std::string uri;
    std::vector<EventConfig> events;
};
// struct Input {
//     Input() = default;
//     unsigned int frame_index_{1};
//     // YoloGPUPtr::Image image;
//     YoloGPUPtr::BoxArray boxarray_;
// };
using Input = std::shared_future<YoloGPUPtr::BoxArray>;
// using ai_callback = MessageCallBackDataInfo;

class EventInfer {
public:
    virtual std::shared_future<std::string> commit(const Input& input) = 0;
    // virtual void set_callback(ai_callback callback) = 0;
    virtual std::string get_uri() const = 0;
    virtual int get_cameraID() const    = 0;
};

std::shared_ptr<EventInfer> create_event(const std::string& raw_data);
}  // namespace Traffic

#endif
