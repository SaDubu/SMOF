#ifndef DEFINE_H
#define DEFINE_H

//sqrt 생략하기 위해 50의 제곱을 사용함.
const float MAX_DIST_SQ = 2500.0f; //(중심을 기준 50픽셀 거리를 max)
const int LIST_SIZE = 30;

#include <vector>

struct ChipInfo {
    cv::Mat image {};
    std::vector<cv::Rect> original_rect {};
};

void add(int* list, int* list_head, int* n) {
    list[*list_head] = *n;
}

int calc_bbox_size(float x1, float y1, float x2, float y2) {
    float diff_x = x2 - x1;
    float diff_y = y2 - y1;

    int area = static_cast<int>(diff_x * diff_y);
    if (area < 0) {
        area *= -1;
    }

    return area;
}

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

// 빈 공간 없이 사용하기 위해 사용함.
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

    float vx = 0.0f;
    float vy = 0.0f;

    bool is_lost = false;

    int bbox_size = 0;
    int missing_count = 0;
};
#if defined(_MSC_VER)
    #pragma pack(pop)
#endif

static int s_tracker_number = 1;
struct Tracker {
    TrackerData data;
    ObjectHistory history;
};

class TrackerVector {
private:
    static const int MAX_CAPACITY = 50;
    Tracker data[MAX_CAPACITY];
    int current_size = 0;

public:
    TrackerVector() : current_size(0) {}

    void erase(int index) {
        if (index < 0 || index >= current_size) return;
        
        data[index] = data[current_size - 1];

        current_size--;
    }

    void cleanup() {
        for (int i = 0; i < current_size; ) {
            if (data[i].data.is_lost || data[i].data.missing_count > 10) {
                erase(i);
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

            new_tr->data.tracker_number = s_tracker_number++;

            return new_tr;
        }
        return nullptr;
    }

    int size() const { return current_size; }
    void clear() { current_size = 0; }
    int capacity() const { return MAX_CAPACITY; }
};

#endif