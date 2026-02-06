#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>

#include <stdlib.h>

#include "MotionDetector.hpp"
#include "LFQSPSC.h"
#include "SharedMemoryManager.hpp"

LockFreeQueueSPSC<std::vector<cv::Mat>> chips_q;

LockFreeQueueSPSC<cv::Mat> raw_q;
LockFreeQueueSPSC<cv::Mat> display_q;
LockFreeQueueSPSC<cv::Mat> motion_q;
LockFreeQueueSPSC<cv::Mat> mask_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> rect_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> bbox_q;
LockFreeQueueSPSC<cv::Mat> final_q;

bool is_running = true;
cv::Size target_size(480, 480);

//실제 오렌지파이 카메라 항목
std::string video_nodes[] = {
    "/dev/video11",
    "/dev/video12",
    "/dev/video13",
    "/dev/video14",
    "/dev/video15",
    "/dev/video16",
    "/dev/video17",
    "/dev/media1",
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3",
    "/dev/video4",
    "/dev/video5",
    "/dev/video6",
    "/dev/video7",
    "/dev/video8",
    "/dev/video9",
    "/dev/video10",
    "/dev/media0",
    "/dev/video18",
    "/dev/video19"
};

std::string CamTest() {
    std::cout << "--- 비디오 장치 스캔 시작 ---" << std::endl;

    int total_nodes = sizeof(video_nodes) / sizeof(video_nodes[0]);
    //pipe format
    std::string pipe = ""; 
    //https://stackoverflow.com/questions/79245401/slow-framerate-from-camera-in-opencvgstreamer-orange-pi5 <- ref
    int i = 0;
    for (i ; i < total_nodes; ++i) {
        pipe = "v4l2src device=" + video_nodes[i] + " is-live=true ! video/x-raw,format=NV12,width=480,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 emit-signals=true sync=false";
        cv::VideoCapture cap;

        cap.open(pipe, cv::CAP_GSTREAMER);

        if (cap.isOpened()) {
            cv::Mat frame;
            cap >> frame; 
            
            if (!frame.empty()) {
                cap.release();
                break;
            }
        }
        cap.release();
        std::cout << "[SKIP] " << video_nodes[i] << std::endl;
    }

    std::cout << "--- 스캔 완료 ---" << std::endl;

    if (i == total_nodes) {
        std::cout << "사용 가능한 카메라 장치를 찾지 못했습니다." << std::endl;
        return "";
    }

    std::cout << "발견된 장치 인덱스: ";
    std::cout << video_nodes[i] << std::endl;

    return pipe;
}

bool mask_moving_area(cv::Mat motion_image, cv::Mat& result) {
    cv::Mat binary, morph;

    cv::threshold(motion_image, binary, 20, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::dilate(binary, morph, kernel);
    cv::erode(morph, morph, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double total_area = 0;
    for (std::vector<cv::Point>& contour : contours) {
        total_area += cv::contourArea(contour);
    }

    if (total_area < 10000) {
        return false;
    }

    result = cv::Mat::zeros(morph.size(), CV_8UC1);
    cv::drawContours(result, contours, -1, cv::Scalar(255), -1);

    return true;
}

//https://blog.naver.com/windrevo/221721329805
std::vector<cv::Rect> merge_boxes(std::vector<cv::Rect>& rects) {
    if (rects.empty()) return {};

    bool changed = true;
    //합쳐지면 처음부터 반복.
    while (changed) {
        changed = false;
        for (int i = 0; i < rects.size(); i++) {
            for (int j = i + 1; j < rects.size(); j++) {
                if ((rects[i] & rects[j]).area() > 0) {
                    rects[i] = rects[i] | rects[j];
                    rects.erase(rects.begin() + j);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    }
    return rects;
}

std::vector<cv::Rect> get_boxes(cv::Mat& mask) {
    std::vector<cv::Rect> result;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rects;
    for (std::vector<cv::Point>& contour : contours) {
        if (cv::contourArea(contour) < 10000) continue; 

        rects.emplace_back(cv::boundingRect(contour));
    }

    //겹치는 박스를 큰 박스로 하나로 정리.
    result = merge_boxes(rects);

    return result;
}

void capture_worker(std::string pipe) {
    cv::VideoCapture cap;
    cap.open(pipe, cv::CAP_GSTREAMER);
    
    while (is_running) {
        cv::Mat frame;

        cap >> frame;
        if (frame.empty()) continue;
        raw_q.Push(frame.clone());
        display_q.Push(frame.clone());
    }
}

void diff_worker(MotionDetector detector) {
    while(is_running) {
        cv::Mat frame, motion_log;

        if (raw_q.Pop(frame)) {
            motion_log = detector.process(frame);
            motion_q.Push(motion_log);
        }
        else {
            std::this_thread::yield();
        }
    }
}

void mask_worker() {
    while (is_running) {
        cv::Mat local_motion, result;
        bool is_next_work = true;

        if (motion_q.Pop(local_motion)) {
            is_next_work = mask_moving_area(local_motion, result);
            if (! is_next_work) {
                std::vector<cv::Rect> e_rect;
                rect_q.Push(e_rect);
                continue;
            }
            mask_q.Push(result);
        }
        else {
            std::this_thread::yield();
        } 
    }
}

void rect_worker() {
    while (is_running) {
        cv::Mat local_mask;

        if (mask_q.Pop(local_mask)) {
            std::vector<cv::Rect> result = get_boxes(local_mask);
            rect_q.Push(result);
        } 
        else {
            std::this_thread::yield();
        }
    }
}

void draw_worker() {
    cv::Mat canvas;
    std::vector<cv::Rect> rects;
    
    while (is_running) {
        if (bbox_q.Pop(rects)) {
            if (display_q.Pop(canvas)) {
                for (cv::Rect& rect : rects) {
                    cv::rectangle(canvas, rect, cv::Scalar(0, 255, 255), 2);
                }

                final_q.Push(canvas);
            }
        }
        else {
            std::this_thread::yield();
        }
    }
}

void crop_worker() {
    cv::Mat frame;
    std::vector<cv::Rect> rects;
    int pad = 10;
    int w_h_pad = pad * 2;

    while (is_running) {
        if (rect_q.Pop(rects)) {
            if (display_q.Pop(frame)) {
                std::vector<cv::Mat> resized_chips;
                int stand_cols = frame.cols;
                int stand_rows = frame.rows;

                for (cv::Rect& rect : rects) {
                    rect.x -= pad;
                    rect.y -= pad;
                    rect.width += w_h_pad;
                    rect.height += w_h_pad;
                    cv::Rect safe_rect = rect & cv::Rect(0, 0, stand_cols, stand_rows);

                    if (safe_rect.width > 0 && safe_rect.height > 0) {
                        cv::Mat roi = frame(safe_rect);
                        cv::Mat resized;

                        cv::resize(roi, resized, target_size, 0, 0, cv::INTER_LINEAR);

                        resized_chips.emplace_back(resized);
                    }
                }
                chips_q.Push(resized_chips);
            }
        }
        else {
            std::this_thread::yield();
        }
    }
}

int main() {  
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    MotionDetector detector;
    SharedMemoryManager smm("yolo_frame", 480, 480);

    std::thread t1(capture_worker, pipe);
    std::thread t2(diff_worker, detector);
    std::thread t3(mask_worker);
    std::thread t4(rect_worker);
    std::thread t5(crop_worker);
    //std::thread t5(draw_worker);

    while (is_running) {
        cv::Mat display_frame;
        std::vector<cv::Mat> frames;
        if (!final_q.Pop(display_frame)) {
            if (chips_q.Pop(frames)) {
                for (cv::Mat& frame : frames) {
                    smm.sendFrame(frame);
                }
                //printf("frame count is {%zu}\n", frames.size());
            }
            continue;
        }

        cv::imshow("boxes", display_frame);
        if (cv::waitKey(1) == 'q') is_running = false;
    }

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join();
    return 0;
}