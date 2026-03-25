#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>

#include <stdlib.h>
#include <sys/stat.h>

#include "define.h"

#include "MotionDetector.hpp"
#include "LFQSPSC.h"
#include "SharedMemoryManager.hpp"
#include "LaborManager.hpp"

struct stat info;

LockFreeQueueSPSC<std::vector<cv::Mat>> chips_q;

LockFreeQueueSPSC<cv::Mat> raw_q;
LockFreeQueueSPSC<cv::Mat> original_q;
LockFreeQueueSPSC<ChipInfo> o_q;
LockFreeQueueSPSC<cv::Mat> display_q;
LockFreeQueueSPSC<cv::Mat> motion_q;
LockFreeQueueSPSC<cv::Mat> mask_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> rect_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> bbox_q;
LockFreeQueueSPSC<cv::Mat> filtered_frame_q;
LockFreeQueueSPSC<cv::Mat> final_q;

LockFreeQueueSPSC<cv::Mat> tunnel;
LockFreeQueueSPSC<std::vector<Detection>> g_detections;
LockFreeQueueSPSC<LockFreeQueueSPSC<std::string>*> file_lists;

LockFreeQueueSPSC<cv::Mat> frames;
LockFreeQueueSPSC<std::vector<cv::Rect>> r_q;

LockFreeQueueSPSC<std::vector<Detection>> yolo_results;
LockFreeQueueSPSC<std::vector<Detection>> filter_results;

std::map<int, std::vector<cv::Point>> path_history;

void sendFrame_g(SharedMemoryManager* p_smm, LockFreeQueueSPSC<cv::Mat>* tunnel, LockFreeQueueSPSC<cv::Mat>* raw_q) {
    cv::Mat frame;
    while (true) {
        if (!tunnel->Pop(frame)) {
            continue;
        }
        //p_smm->sendFrame(frame);

        raw_q->Push(frame.clone());
    }
}

void drawBoxes(cv::Mat& image, const std::vector<cv::Rect>& frame_rect) {
    for (const auto& rect : frame_rect) {
        cv::rectangle(image, 
                      rect, 
                      cv::Scalar(0, 255, 0),
                      2,                   
                      cv::LINE_8);         
        
    }
}

int run() {
    /*
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    */
    std::string pipe = "20";
    cv::Mat raw_image;
    cv::Mat* send_image = nullptr;
    cv::Mat display;
    size_t count = 0;

    bool is_running = true;

    MotionDetector detector;
    TrackerVector tracker_vector;
    LaborManager lm(&is_running);
    SharedMemoryManager smm("yolo_frame", 640, 640);
    smm.setChipsNum(0);
    std::vector<Detection>* recive_output;
    LockFreeQueueSPSC<std::vector<Detection>> detections;
    std::vector<cv::Rect> frame_rect;

    std::thread t1([&]() {
        lm.capture_worker(pipe, raw_q, display_q);
    });
    std::thread t2(sendFrame_g, &smm, &tunnel, &raw_q);
    std::thread t3([&]() {
        lm.diff_worker(detector, raw_q, motion_q);
    });
    std::thread t4([&]() {
        lm.mask_worker(motion_q, rect_q, mask_q);
    });
    std::thread t5([&]() {
        lm.rect_worker(mask_q, rect_q);
    });
    std::thread t6([&]() {
        lm.track_worker(g_detections, &tracker_vector);
    });
    int size = 0;
    while(true) {
        if (!display_q.Pop(raw_image)) {
            send_image = nullptr;
            raw_image.release();
            continue;
        }
        tunnel.Push(raw_image.clone());

        recive_output = smm.receiveYoloResult();
        if (recive_output == nullptr) {
            continue;
        }
        rect_q.Pop(frame_rect);
        std::vector<Detection> temp = *recive_output;
        keepBestDetectionByCenter(temp, frame_rect);
        size = temp.size();
        detections.Push(temp); 

        mut_draw_tracker_visualization(raw_image, tracker_vector, &size, &path_history);

        drawBoxes(raw_image, frame_rect);
        ++count;
        std::string save_path = cv::format("/test_object_rule/try_3/image/%ld.jpg", count);
        cv::imwrite(save_path, raw_image);
        continue;

        if (size > 0) {
            final_q.Push(raw_image);
        }

        //draw_box(&raw_image, &detections, &final_q, &is_running);

        if (!final_q.Pop(display)) {
            display.release();
            continue;
        }
        else {
            if (!display.empty()) {
                ++count;
                //std::string save_path = cv::format("/test_object_rule/try_3/image/%ld.jpg", count);  
                //cv::imwrite(save_path, display);
                cv::imshow("hi", display);
                cv::waitKey(1);
            }
        }

    }

    t1.join();
    return 0;
}

void ImageFileLoader(std::string path, LockFreeQueueSPSC<cv::Mat>& display_q, bool& m_is_running) { 
    while (m_is_running) {
        std::vector<cv::String> fn;
        cv::glob(path + "/*.jpeg", fn, false);

        for (const auto& file_path : fn) {
            cv::Mat frame = cv::imread(file_path);
                
            if (!frame.empty()) {
                cv::resize(frame, frame, cv::Size(640, 480));
                display_q.Push(frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
}

void saveDetectionLog(std::vector<Detection>* dets) {
    if (dets == nullptr || dets->empty()) return;

    std::ofstream logFile("snack_test_log.csv", std::ios::app);
    
    if (logFile.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        for (Detection& det : *dets) {
            logFile << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") 
                    << "." << std::setfill('0') << std::setw(3) << ms.count() << ","
                    << (int)det.class_id << "," 
                    << std::fixed << std::setprecision(4) << det.confidence << "\n";
        }
        
        logFile.close();
    }
}

int run_1() {
    /*
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    */
    //std::string pipe = "/home/orangepi/Projects/SMOF/scenario_frame/scenario_21";
    std::string pipe = "20";
    cv::Mat raw_image;
    cv::Mat* send_image = nullptr;
    cv::Mat display;
    size_t count = 0;

    bool is_running = true;

    MotionDetector detector;
    TrackerVector tracker_vector;
    LaborManager lm(&is_running);
    SharedMemoryManager smm("yolo_frame", 640, 480);
    smm.setChipsNum(0);
    std::vector<Detection>* recive_output;

    std::thread t1(ImageFileLoader, pipe, std::ref(display_q), std::ref(is_running));
    std::thread t2(sendFrame_g, &smm, &tunnel, &raw_q);
    while(true) {
        if (!display_q.Pop(raw_image)) {
            send_image = nullptr;
            raw_image.release();
            continue;
        }

        tunnel.Push(raw_image);

        recive_output = smm.receiveYoloResult();
        saveDetectionLog(recive_output);
        if (recive_output == nullptr) {
            continue;
        }      
    }
    t1.join(); t2.join();
    return 0;
}

void draw_tracker_visualization(cv::Mat& frame, TrackerVector& trackers) {

    cv::Scalar zone_color(100, 100, 100);
    
    cv::line(frame, cv::Point(X_MIN, 0), cv::Point(X_MIN, frame.rows), zone_color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(X_MAX, 0), cv::Point(X_MAX, frame.rows), zone_color, 1, cv::LINE_AA);
    
    cv::putText(frame, "ZONE: " + std::to_string(X_MIN) + " ~ " + std::to_string(X_MAX),
                cv::Point(X_MIN + 5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1, cv::LINE_AA);
    for (int i = 0; i < trackers.size(); ++i) {
        TrackerData& d = trackers[i].data;

        int id = d.tracker_number;
        cv::Scalar unique_color(
            (id * 77) % 255,   // Blue
            (id * 135) % 255,  // Green
            (id * 213) % 255   // Red
        );

        if (d.missing_count > 0) {
            cv::Scalar red_color(0, 0, 255);
            cv::Rect rect(cv::Point(d.past_x1, d.past_y1), cv::Point(d.past_x2, d.past_y2));
            cv::rectangle(frame, rect, red_color, 2);

            std::string label = "ID: " + std::to_string(id) + 
                                " (Miss)";
            cv::putText(frame, label, cv::Point(d.past_x1, d.past_y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1);
        }

        cv::Point center(d.past_cx, d.past_cy);
        cv::Point velocity_tip(d.past_cx + d.vx * 5, d.past_cy + d.vy * 5);
        
        cv::arrowedLine(frame, center, velocity_tip, unique_color, 2, 8, 0, 0.3);

        cv::circle(frame, center, 3, unique_color, -1);
    }
}

int run_2() {
    if (stat(record_dir.c_str(), &info) != 0) {
        if (mkdir(record_dir.c_str(), 0755) == -1) {
            std::cerr << "Error: Could not create directory " << record_dir << std::endl;
        }
    }

    std::string pipe = "20";
    cv::Mat frame;
    cv::Mat display;
    size_t count = 0;

    bool is_runnig = true;

    MotionDetector detector;
    TrackerVector tracker_vector;
    LaborManager lm(&is_runnig);
    SharedMemoryManager smm("yolo_frame", 640, 480);
    smm.setChipsNum(0);

    std::string window_name = "test";

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    std::thread t1([&]() {
        lm.capture_worker(pipe, raw_q, display_q);
    });
    std::thread t2(sendFrame_g, &smm, &tunnel, &raw_q);
    std::thread t3([&]() {
        lm.dmr_worker(detector, raw_q, motion_q, rect_q);
    });
    std::thread t4([&]() {
        lm.RGB_draw_save_worker(rect_q, motion_q, final_q, file_lists);
    });
    std::thread t5([&]() {
        lm.get_image_move_area(file_lists, frames, r_q);
    });
    std::thread t6([&]() {
        lm.yolo_worker(smm, frames, yolo_results);
    });
    std::thread t7([&]() {
        lm.filter_worker(yolo_results, r_q, filter_results);
    });
    std::thread t8([&]() {
        lm.track_worker(filter_results, &tracker_vector);
    });

    while (is_runnig) {
        cv::waitKey(1);
        if (!display_q.Pop(frame)) {
            frame.release();
            continue;
        }
        tunnel.Push(frame);

        if (!final_q.Pop(display)) {
            display.release();
            continue;
        }

        //display_tracker_monitor(tracker_vector);
        //draw_tracker_visualization(display, tracker_vector);
        cv::resize(display, display, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR);
        cv::imshow(window_name, display);
    }
    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join();
    return 0;
}

int main() {
    run_2();
    return 0;
}