/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-01-13 13:06:23
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 14:02:58
 *************************************************************************************/

#include <stdio.h>
#include <string.h>

#include <common/ilogger.hpp>
#include <functional>

int app_yolo();
int app_json();
int app_bus();
int app_test();
int app_yolopose();
int app_plate();
int main(int argc, char **argv) {
    const char *method = "yolo";
    if (argc > 1) {
        method = argv[1];
    }

    if (strcmp(method, "plate") == 0) {
        app_plate();
    } else if (strcmp(method, "yolo") == 0) {
        app_yolo();
    } else if (strcmp(method, "json") == 0) {
        app_json();
    } else if (strcmp(method, "bus") == 0) {
        app_bus();
    } else if (strcmp(method, "test") == 0) {
        app_test();
    } else {
        app_yolo();
    }
    return 0;
}
