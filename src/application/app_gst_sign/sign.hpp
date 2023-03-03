#pragma once
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace Sign {
using namespace std;

class Solution {
public:
    virtual void commit() = 0;
};
shared_ptr<Solution> create_solution(const string &det_name, bool use_device_frame = true);

}  // namespace Sign