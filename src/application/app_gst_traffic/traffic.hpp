#ifndef __TRAFFIC_HPP__
#define __TRAFFIC_HPP__
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace Traffic {
struct Image {
    const void* bgrptr = nullptr;
    int width = 0, height = 0;
    int camera_id = 0;
    int device_id = -1;  // -1: cpuptr;

    Image() = default;
    Image(const void* bgrptr, int width, int height, int camera_id = 0, int device_id = -1)
        : bgrptr(bgrptr), width(width), height(height), camera_id(camera_id), device_id(device_id) {}
};
class Solution {
public:
    // 根据配置启用一路摄像头
    virtual bool make_view(const std::string& raw_data, size_t timeout = 100)                      = 0;
    virtual std::shared_future<std::string> commit(const Image& image)                             = 0;
    virtual std::vector<std::shared_future<std::string>> commits(const std::vector<Image>& images) = 0;
};
std::shared_ptr<Solution> create_solution(const std::string& det_name);
}  // namespace Traffic
#endif
