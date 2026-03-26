#include "LaborManager.hpp"
#include "MotionDetector.hpp"
#include "LFQSPSC.h"
#include "SharedMemoryManager.hpp"

LaborManager::LaborManager(bool* is_running_ptr) : m_is_running(is_running_ptr) {

}

bool LaborManager::mask_moving_area(cv::Mat& motion_image, cv::Mat& result) {
    cv::Mat binary, morph;

    cv::threshold(motion_image, binary, 50, 255, cv::THRESH_BINARY);

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
    //cap.open(pipe, cv::CAP_GSTREAMER);
    cap.open(std::stoi(pipe), cv::CAP_V4L2);

    while (m_is_running) {
        cv::Mat frame;

        cap >> frame;
        if (frame.empty()) continue;
        display_q.Push(frame.clone());
    }
}

void LaborManager::diff_worker(MotionDetector& detector, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& motion_q) {
    while(m_is_running) {
        cv::Mat frame, motion_log, result;

        if (raw_q.Pop(frame)) {
            motion_log = detector.process(frame);
            motion_q.Push(motion_log);
        }
        else {
            std::this_thread::yield();
        }
    }
}

void LaborManager::dmr_worker(MotionDetector& detector, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& motion_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q) {
    while(m_is_running) {
        cv::Mat frame, motion_log, result;

        if (raw_q.Pop(frame)) {
            motion_q.Push(frame.clone());
            motion_log = detector.process(frame);
            mask_moving_area(motion_log, result);
            std::vector<cv::Rect> rects = get_boxes(result);
            rect_q.Push(rects);
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
            /*
            if (! is_next_work) {
                printf("mask_worker Error, No result");
            }
            */
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

void LaborManager::distribution_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& in_rect_q, LockFreeQueueSPSC<cv::Mat>& in_mat_q, 
        LockFreeQueueSPSC<std::vector<cv::Rect>>& out_rect_q, LockFreeQueueSPSC<cv::Mat>& out_mat_q,
        LockFreeQueueSPSC<LockFreeQueueSPSC<std::string>*>& file_lists) {
    cv::Mat mat;
    std::vector<cv::Rect> rects;
    LockFreeQueueSPSC<std::string>* file_list;

    size_t count = 0;

    while (m_is_running) {
        if (in_rect_q.Pop(rects)) {
            if (in_mat_q.Pop(mat)) {
                if (!rects.empty()) {
                    move_frame_save(mat, rects);
                }
                else {
                    ++count;
                    if (count == 90) { //약 3초 이내
                        file_list = get_file_list();
                        file_lists.Push(file_list);
                        count = 0;
                    }
                }
                std::vector<cv::Rect> out_rects = rects;
                out_rect_q.Push(out_rects);
                out_mat_q.Push(mat.clone());
            }
        }
    }
}   

void LaborManager::RGB_draw_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& bbox_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& final_q) {
    cv::Mat canvas;
    std::vector<cv::Rect> rects;
    
    while (m_is_running) {
        if (bbox_q.Pop(rects)) {
            if (display_q.Pop(canvas)) {
                int w = canvas.cols;
                int h = canvas.rows;
                int one_third = w / 3;
                int two_thirds = (w / 3) * 2;

                cv::line(canvas, cv::Point(one_third, 0), cv::Point(one_third, h), cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
                cv::line(canvas, cv::Point(two_thirds, 0), cv::Point(two_thirds, h), cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

                for (cv::Rect& rect : rects) {
                    cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    cv::Scalar color;

                    if (center.x < one_third) {
                        color = cv::Scalar(0, 0, 255); 
                    } 
                    else if (center.x < two_thirds) {
                        color = cv::Scalar(0, 255, 0);
                    } 
                    else {
                        color = cv::Scalar(255, 0, 0); 
                    }

                    cv::circle(canvas, center, 5, color, -1, cv::LINE_AA);
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
// 이 경우에 문제는 하나의 박스로 되지 않는 경우에는 2개로 나누어지게 된다.
// python에서 추론한 object의 갯수를 보낼 수 있도록 설계하였기 때문에 각 chip(움직이는 영역을 잘라둔 것)에서 추론된
// object의 class 중 가장 높은 점수를 기록한 추론 정보만 넘기도록 하면 count를 chip의 갯수에 맞게 설계할 수 있을 것으로 보임.
void LaborManager::crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<std::vector<cv::Mat>>& chips_q, LockFreeQueueSPSC<ChipInfo>& o_q) {
    cv::Mat frame;
    std::vector<cv::Rect> rects;
    int pad = 10;
    int w_h_pad = pad * 2;
    cv::Size target_size(480, 480);

    while (m_is_running) {
        if (rect_q.Pop(rects)) {
            if (display_q.Pop(frame)) {
                ChipInfo chip_info;
                std::vector<cv::Mat> resized_chips;
                std::vector<cv::Rect> safe_rects;
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
                        safe_rects.emplace_back(safe_rect);
                    }
                }
                chips_q.Push(resized_chips);
                chip_info.image = frame.clone();
                chip_info.original_rect = safe_rects;
                o_q.Push(chip_info);
            }
        }
        else {
            std::this_thread::yield();
        }
    }
}

//움직이는 부분만 값 유지.
// * 문제점 : 하얀 배경으로 학습을 시켜서 그런지 움직이는 물체와 그 주변부 자체를 하나의 object로 인식을 함.
//           그래서 제대로 인식을 해주지 않음. 
void LaborManager::new_crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& filtered_frame_q) {
    cv::Mat frame;
    std::vector<cv::Rect> rects;
    int pad = 10;
    int w_h_pad = pad * 2;
    
    while (m_is_running) {
        if (rect_q.Pop(rects)) {
            if (display_q.Pop(frame)) {
                cv::Mat result(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
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

//tracker의 현 frame 업데이트 진행.
void tracker_update(TrackerVector* trackers) {
    for (int i = 0; i < trackers->size(); ++i) {
        Tracker& t = (*trackers)[i];
        if (!t.data.checked_in) {
            t.data.checked_in = check_x_is_here((int)t.data.past_cx); //영역 내부로 들어왔는지 여기서 확인.
        }
        
        t.data.past_cx += t.data.vx;
        t.data.past_cy += t.data.vy;
        
        t.data.past_x1 += t.data.vx;
        t.data.past_x2 += t.data.vx;
        t.data.past_y1 += t.data.vy;
        t.data.past_y2 += t.data.vy;

        ++t.data.missing_count; 
    }
}

void tracker_match(Detection* object, TrackerVector* trackers) {
    float cx = (object->x1 + object->x2) * 0.5f;
    float cy = (object->y1 + object->y2) * 0.5f;

    int best_match_idx = -1;
    float min_dist_sq = 999999.0f;
    float best_iou = -1.0f;

    for (int i = 0; i < trackers->size(); ++i) {
        Tracker& t = (*trackers)[i];
        if (object->class_id == -2.0f) {
            object->class_id = *t.history.get_infer_class();
        }
        
        int conditions_met = 0;

        float dx = t.data.past_cx - cx;
        float dy = t.data.past_cy - cy;
        float dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= MAX_DIST_SQ) {
            conditions_met++;
        }

        int area_object = calc_bbox_size(object->x1, object->y1, object->x2, object->y2);
        int area_past = *t.history.get_bbox_average();

        float iou = calculate_iou(object->x1, object->y1, object->x2, object->y2, area_object,
                                  t.data.past_x1, t.data.past_y1, t.data.past_x2, t.data.past_y2, area_past);
        
        if (iou >= IOU_THRESHOLD) {
            conditions_met++;
        }

        float ratio = calculate_2Box_size_ratio(area_object, area_past);

        if (ratio >= IOU_THRESHOLD) {
            conditions_met++;
        }

        if (object->class_id == *t.history.get_infer_class()) {
            conditions_met++;
        }

        if (conditions_met >= 2) {
            if (iou > best_iou) {
                best_iou = iou;
                best_match_idx = i;
            }
        }
    }

    if (best_match_idx != -1) {
        Tracker& matched_tr = (*trackers)[best_match_idx];

        //printf("Matched ID: %d | Diff: %.2f, %.2f\n", matched_tr.data.tracker_number, cx - matched_tr.data.past_cx, cy - matched_tr.data.past_cy);
        
        matched_tr.data.vx = cx - (matched_tr.data.past_cx - matched_tr.data.vx);
        matched_tr.data.vy = cy - (matched_tr.data.past_cy - matched_tr.data.vy);
        
        matched_tr.data.past_cx = cx;
        matched_tr.data.past_cy = cy;
        
        matched_tr.data.past_x1 = object->x1;
        matched_tr.data.past_y1 = object->y1;
        matched_tr.data.past_x2 = object->x2;
        matched_tr.data.past_y2 = object->y2;

        matched_tr.data.missing_count = 0;
        int area_object = calc_bbox_size(object->x1, object->y1, object->x2, object->y2);
        matched_tr.history.add(static_cast<int>(object->class_id), area_object);
    }
    else {
        // 매칭 실패 시 새로운 Tracker 생성
        Tracker* new_tr = trackers->emit_back();
        
        if (new_tr != nullptr) {
            new_tr->data.past_cx = cx;
            new_tr->data.past_cy = cy;
            
            new_tr->data.past_x1 = object->x1;
            new_tr->data.past_y1 = object->y1;
            new_tr->data.past_x2 = object->x2;
            new_tr->data.past_y2 = object->y2;
            
            new_tr->data.vx = 0.0f;
            new_tr->data.vy = 0.0f;
            new_tr->data.missing_count = 0;
            int area_object = calc_bbox_size(object->x1, object->y1, object->x2, object->y2);
            new_tr->history.add(static_cast<int>(object->class_id), area_object);
        }
    }

}

void LaborManager::track_worker(LockFreeQueueSPSC<std::vector<Detection>>& objects_q, TrackerVector* trackers) {
    std::vector<Detection> objects;

    Detection* object;

    size_t count = 0;

    while (*m_is_running) {
        if (!objects_q.Pop(objects)) {
            ++count;
            if (count >= 15) {
                trackers->clear();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        count = 0;

        if (trackers->size() > 0) {
            tracker_update(trackers);
        }

        for (int i = 0; i < objects.size(); ++i) {
            tracker_match(&objects[i], trackers);
        }

        trackers->cleanup();
    }
}

void LaborManager::yolo_worker(SharedMemoryManager& smm, LockFreeQueueSPSC<cv::Mat>& frames, LockFreeQueueSPSC<std::vector<Detection>>& yolo_results) {
    std::vector<Detection>* receive_output;
    cv::Mat frame;

    while (m_is_running) {
        if (!frames.Pop(frame)) {
            continue;
        }
        if (frame.empty()) {
            continue;
        }
        smm.sendFrame(frame);

        receive_output = smm.receiveYoloResult();
        if (receive_output == nullptr) {
            continue;
        }

        std::vector<Detection> dets = *receive_output;

        yolo_results.Push(dets);
    }
}

void LaborManager::filter_worker(LockFreeQueueSPSC<std::vector<Detection>>& yolo_results, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<std::vector<Detection>>& filter_results) {
    std::vector<cv::Rect> rects;
    std::vector<Detection> yolo_result;

    while (m_is_running) {
        if (!yolo_results.Pop(yolo_result)) {
            continue;
        }

        if (!rect_q.Pop(rects)) {
            continue;
        }

        keepBestDetectionByCenter(yolo_result, rects);
        filter_results.Push(yolo_result);
    }
}

void LaborManager::get_image_move_area(LockFreeQueueSPSC<LockFreeQueueSPSC<std::string>*>& file_lists, LockFreeQueueSPSC<cv::Mat>& frames, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q) {
    LockFreeQueueSPSC<std::string>* file_list = nullptr;
    std::string file_path = "";
    std::string image_path = "";
    std::string move_area_path = "";
    std::vector<cv::Rect> rects;
    cv::Mat frame;
    while (m_is_running) {
        if (!file_lists.Pop(file_list)) {
            continue;
        }

        if (file_list == nullptr) {
            continue;
        }

        while (true) {
            if (!file_list->Pop(file_path)) {
                break;
            }

            move_area_path = file_path + ".txt";
            image_path = file_path + ".jpeg";

            rects = get_target_regions_from_file(move_area_path);
            frame = get_target_image_from_file(image_path);

            rect_q.Push(rects);
            frames.Push(frame);
        }

        delete file_list;
        file_list = nullptr;
    }
}