#ifndef DEFINE_H
#define DEFINE_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <unistd.h>
#include "LFQSPSC.h"

//sqrt 생략하기 위해 50의 제곱을 사용함.
const float MAX_DIST_SQ = 10000.0f; //(중심을 기준 100픽셀 거리를 max)
const int LIST_SIZE = 30;
const float IOU_THRESHOLD = 0.5f;
const float EPSILON = 1e-4f;
const int MAX_MISS_COUNT = 30;

//물체가 들어와야하는 x 영역
const int X_MIN = 213;
const int X_MAX = 426;

static const char* SNACK_LIST[] = {
    "pepero",               
    "lotte_sand",           
    "butter_ring",           
    "oreo_thins",            
    "zec",                   
    "pringles",             
    "jagabee",               
    "spicy_shrimp_cracker", 
    "cheetos",             
    "pocachip"             
};

struct ChipInfo {
    cv::Mat image {};
    std::vector<cv::Rect> original_rect {};
};

void add(int* list, int* list_head, int* n);

int calc_bbox_size(float x1, float y1, float x2, float y2);

float calculate_iou(float x1_a, float y1_a, float x2_a, float y2_a, int area_a,
                    float x1_b, float y1_b, float x2_b, float y2_b, int area_b);

float calculate_2Box_size_ratio(int area_a, int area_b);

extern std::string record_dir;

bool check_x_is_here(int object_cx);

void move_frame_save(cv::Mat& frame, std::vector<cv::Rect>& move_area);

std::vector<cv::Rect> get_target_regions_from_file(std::string& filepath);

cv::Mat get_target_image_from_file(std::string& filepath);

LockFreeQueueSPSC<std::string>* get_file_list();

void delete_target_file(std::string& filepath);

std::string CamTest();

//std::string CamTest();

class ObjectHistory {
private:
    int class_history[LIST_SIZE] = {0};
    int bbox_history[LIST_SIZE] = {0};

    float bbox_average = 0.0f;
    int infer_class = -1;

    int head = 0;
    int count = 0;

public:
    ObjectHistory() : head(0), count(0) {}

    void add(int class_id, int bbox_size) {
        ::add(class_history, &head, &class_id);
        ::add(bbox_history, &head, &bbox_size);
        head = (head + 1) % LIST_SIZE;
        if (count < LIST_SIZE) ++count;
    }

    int* get_infer_class() {
        if (count == 0) return &infer_class;

        infer_class = class_history[(head + LIST_SIZE - 1) % LIST_SIZE];
        int max_count = 0;

        for (int i = 0; i < count; ++i) {
            int current = class_history[i];
            
            bool is_seen = false;
            for (int j = 0; j < i; ++j) {
                if (class_history[j] == current) {
                    is_seen = true;
                    break;
                }
            }
            if (is_seen) continue;

            int current_count = 0;
            for (int j = 0; j < count; ++j) {
                if (class_history[j] == current) ++current_count;
            }

            if (current_count > max_count) {
                max_count = current_count;
                infer_class = current;
            }
        }
        return &infer_class;
    }

    float* get_bbox_average() {
        if (count == 0) return &bbox_average;

        float sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            sum += bbox_history[i];
        }

        bbox_average = sum / (float)count;

        return &bbox_average;
    }
};

//값을 float로 보내기 때문임.
struct Detection {
    float x1, y1, x2, y2, confidence, class_id;
}; 

#if defined(_MSC_VER) // window
    #define PACKED_STRUCT struct
    #pragma pack(push, 1)
#elif defined(__GNUC__) //linux
    #define PACKED_STRUCT struct __attribute__((packed))
#else
    #define PACKED_STRUCT struct
#endif
struct TrackerData {
    int tracker_number;
    float past_cx, past_cy;
    float past_x1, past_y1, past_x2, past_y2;

    float vx = 0.0f;
    float vy = 0.0f;

    bool checked_in = false;

    int bbox_size = 0;
    int missing_count = 0;
};
#if defined(_MSC_VER)
    #pragma pack(pop)
#endif

struct Tracker {
    TrackerData data;
    ObjectHistory history;
};

class TrackerVector {
private:
    static const int MAX_CAPACITY = 50;
    Tracker data[MAX_CAPACITY];
    int current_size = 0;
    int next_id = 1;

    size_t how_many_erase = 0;

public:
    TrackerVector() : current_size(0), next_id(1) {}

    void erase(int index) {
        if (index < 0 || index >= current_size) return;
        
        data[index] = data[current_size - 1];

        current_size--;
    }

    void alarm(int index) {
        int class_id = *data[index].history.get_infer_class(); 

        if (class_id < 0) {
            return;
        }
        if (data[index].data.checked_in) {
            return;
        }
        printf("Miss Scan Name: %s\n", SNACK_LIST[class_id]);
    }

    void cleanup() {
        int i = 0;
        while (i < current_size) {
            if (data[i].data.missing_count > MAX_MISS_COUNT) {
                alarm(i);
                erase(i);
                ++how_many_erase; 
            } else {
                ++i;
            }
        }
    }

    Tracker& operator[](int index) {
        return data[index];
    }

    //여기서 tracker number(id) 증가 시킴.
    Tracker* emit_back() {
        if (current_size < MAX_CAPACITY) {
            Tracker* new_tr = &data[current_size++];

            *new_tr = Tracker();

            new_tr->data.tracker_number = next_id++;

            return new_tr;
        }
        return nullptr;
    }

    void clear() { 
        int i = 0;
        while (i < current_size) {
            alarm(i);
            ++i;
        }
        current_size = 0; 
    }

    int size() const { return current_size; }
    int capacity() const { return MAX_CAPACITY; }
    size_t* get_how_many_erase() { return &how_many_erase; }
};

void keepBestDetectionByCenter(std::vector<Detection>& detections, std::vector<cv::Rect>& target_regions);

void mut_draw_tracker_visualization(cv::Mat& frame, TrackerVector& trackers, int* rejected_detections_num, std::map<int, std::vector<cv::Point>>* path_history);

void display_tracker_monitor(TrackerVector& trackers);

#endif