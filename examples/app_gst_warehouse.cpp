#include <math.h>

#include <initializer_list>
#include <iostream>

#include "app_gst_warehouse/warehouse.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

static void callback(int callbackType, void *img, char *data, int datalen) {
    ;
}

void test_warehouse() {
    // debug
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    // iLogger::rmtree("imgs_callback");
    // iLogger::mkdir("imgs_callback");
    std::string det_name  = "yolov6s_qa";
    std::string pose_name = "yolov5s_pose";
    std::string gcn_name  = "";
    std::vector<std::string> uris{"rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101",
                                  "rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101"};

    auto pipeline = Warehouse::create_solution(det_name, "", gcn_name);

    if (pipeline == nullptr) {
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    pipeline->set_callback(callback);
    for (auto uri : uris) {
        pipeline->make_view(uri);
    }

    auto current_uris = pipeline->get_uris();
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }
    iLogger::sleep(50000000);
    pipeline->stop();
}

int app_warehouse() {
    test_warehouse();
    return 0;
}