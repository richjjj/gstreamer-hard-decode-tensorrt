#ifndef __WAREHOUSEV2_HPP__
#define __WAREHOUSEV2_HPP__
#include <memory>
#include <string>
#include <vector>
namespace WarehouseV2 {
struct Box {
    float left, top, right, bottom, confidence;
    int class_label;
    int id;  // track id

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label, int id)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label),
          id(id) {}
};
typedef std::vector<Box> BoxArray;

struct Image {
    const void* bgrptr = nullptr;
    int width = 0, height = 0;
    int camra_id  = 0;
    int device_id = -1;  // -1: cpuptr;

    Image() = default;
    Image(const void* bgrptr, int width, int height, int device_id = -1)
        : bgrptr(bgrptr), width(width), height(height), device_id(device_id) {}
};

class Solution {
public:
    virtual std::string commit(const Image& image)                             = 0;
    virtual std::vector<std::string> commits(const std::vector<Image>& images) = 0;
};
std::shared_ptr<Solution> create_solution(const std::string& det_name);

}  // namespace WarehouseV2

#endif