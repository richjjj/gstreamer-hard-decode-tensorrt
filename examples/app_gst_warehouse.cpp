#include <math.h>

#include <initializer_list>
#include <iostream>

#include "app_gst_warehouse/warehouse.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

static void callback(int callbackType, void *img, char *data, int datalen) {}

void test_warehouse() {
    // debug
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    // iLogger::rmtree("imgs_callback");
    // iLogger::mkdir("imgs_callback");
    std::string det_name  = "yolov6s_qa";
    std::string pose_name = "yolov5s_pose";
    std::string gcn_name  = "";
    // std::vector<std::string> uris{"exp/39.mp4", "exp/37.mp4", "exp/38.mp4",
    //                               "exp/37.mp4", "exp/38.mp4", "rtsp://192.168.170.109:554/live/streamperson"};
    std::vector<std::string> uris{"rtsp://admin:admin123@192.168.170.109:580/cam/realmonitor?channel=4&subtype=0",
                                  "rtsp://admin:admin123@192.168.170.109:580/cam/realmonitor?channel=6&subtype=0"};

    auto pipeline = Warehouse::create_solution(det_name, pose_name, gcn_name);

    if (pipeline == nullptr) {
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    //
    pipeline->set_callback(callback);
    // auto out = pipeline->make_views(uris);
    for (auto uri : uris) {
        pipeline->make_view(uri);
    }

    // for (auto x : out) {
    //     std::cout << x << std::endl;
    // }

    auto current_uris = pipeline->get_uris();
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }
    // test disconnet
    // pipeline->disconnect_view("rtsp://192.168.170.109:554/live/streamperson6");
    current_uris = pipeline->get_uris();
    std::cout << "after disconnect_view(rtsp://192.168.170.109:554/live/streamperson6): "
              << "\n";
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }
}

int app_warehouse() {
    test_warehouse();
    return 0;
}