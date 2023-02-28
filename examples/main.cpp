
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
int app_gstdecode();
int app_warehouse();
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
    } else if (strcmp(method, "yolopose") == 0) {
        app_yolopose();
    } else if (strcmp(method, "gstdecode") == 0) {
        app_gstdecode();
    } else if (strcmp(method, "warehouse") == 0) {
        app_warehouse();
    } else {
        app_yolo();
    }
    return 0;
}
