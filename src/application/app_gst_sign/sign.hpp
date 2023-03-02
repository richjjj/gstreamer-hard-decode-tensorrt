#pragma once
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace Sign {
using namespace std;
using ai_callback = function<void(int callbackType, void *image, char *data, int datalen)>;
// using ai_callback = CallBackDataInfo;

class Solution {
public:
    // Here, raw_data means json_data.dump()
    // Not thread safe
    virtual bool make_view(const string &raw_data, size_t timeout = 100) = 0;
    virtual void set_callback(ai_callback callback)                      = 0;
    // stop 所有视频流
    virtual void stop() = 0;
    // 停止指定视频流，dis_uri 为rtsp流地址
    virtual void disconnect_view(const string &dis_uri) = 0;
    // 获取当前所有的视频流
    virtual vector<string> get_uris() const = 0;
};
shared_ptr<Solution> create_solution(const string &det_name, bool use_device_frame = true);

}  // namespace Sign