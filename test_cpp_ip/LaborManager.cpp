#include "LaborManager.hpp"
#include "MotionDetector.hpp"
#include "LFQSPSC.h"

LaborManager::LaborManager(bool* is_running_ptr) : m_is_running(is_running_ptr) {

}

bool LaborManager::mask_moving_area(cv::Mat& motion_image, cv::Mat& result) {
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
std::vector<cv::Rect> LaborManager::merge_boxes(std::vector<cv::Rect>& rects) {
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

std::vector<cv::Rect> LaborManager::get_boxes(cv::Mat& mask) {
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

void LaborManager::capture_worker(std::string& pipe, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& display_q) {
    cv::VideoCapture cap;
    cap.open(pipe, cv::CAP_GSTREAMER);
    
    while (m_is_running) {
        cv::Mat frame;

        cap >> frame;
        if (frame.empty()) continue;
        raw_q.Push(frame.clone());
        display_q.Push(frame.clone());
    }
}

void LaborManager::diff_worker(MotionDetector& detector, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& motion_q) {
    while(m_is_running) {
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

void LaborManager::mask_worker(LockFreeQueueSPSC<cv::Mat>& motion_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& mask_q) {
    while (m_is_running) {
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

void LaborManager::rect_worker(LockFreeQueueSPSC<cv::Mat>& mask_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q) {
    while (m_is_running) {
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

void LaborManager::draw_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& bbox_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& final_q) {
    cv::Mat canvas;
    std::vector<cv::Rect> rects;
    
    while (m_is_running) {
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

//resize로 진행을 하고 있는 부분
void LaborManager::crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<std::vector<cv::Mat>>& chips_q) {
    cv::Mat frame;
    std::vector<cv::Rect> rects;
    int pad = 10;
    int w_h_pad = pad * 2;
    cv::Size target_size(480, 480);

    while (m_is_running) {
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

//움직이는 부분만 값 유지 나머지는 검은색 처리
void LaborManager::new_crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& filtered_frame_q) {
    cv::Mat frame;
    std::vector<cv::Rect> rects;
    int pad = 10;
    int w_h_pad = pad * 2;
    
    while (m_is_running) {
        if (rect_q.Pop(rects)) {
            if (display_q.Pop(frame)) {
                cv::Mat result = cv::Mat::zeros(frame.size(), frame.type());
                int stand_cols = frame.cols;
                int stand_rows = frame.rows;

                for (cv::Rect& rect : rects) {
                    rect.x -= pad;
                    rect.y -= pad;
                    rect.width += w_h_pad;
                    rect.height += w_h_pad;
                    cv::Rect safe_rect = rect & cv::Rect(0, 0, stand_cols, stand_rows);

                    if (safe_rect.width > 0 && safe_rect.height > 0) {
                        frame(safe_rect).copyTo(result(safe_rect));
                    }
                }

                filtered_frame_q.Push(result);
            }
        }
        else {
            std::this_thread::yield();
        }
    }
}