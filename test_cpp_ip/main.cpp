#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include <stdlib.h>

#include "define.h"

#include "MotionDetector.hpp"
#include "LFQSPSC.h"
#include "SharedMemoryManager.hpp"
#include "LaborManager.hpp"

LockFreeQueueSPSC<std::vector<cv::Mat>> chips_q;

LockFreeQueueSPSC<cv::Mat> raw_q;
LockFreeQueueSPSC<cv::Mat> display_q;
LockFreeQueueSPSC<cv::Mat> motion_q;
LockFreeQueueSPSC<cv::Mat> mask_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> rect_q;
LockFreeQueueSPSC<std::vector<cv::Rect>> bbox_q;
LockFreeQueueSPSC<cv::Mat> filtered_frame_q;
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

namespace TestScope {
    struct s_label {
        int class_id;
        float x1, y1, x2, y2;
    };

    int get_images(std::string path, LockFreeQueueSPSC<cv::Mat>* raw_q, LockFreeQueueSPSC<cv::Mat>* display_q) {
        std::vector<std::string> paths;

        cv::glob(path, paths, false);

        size_t count = 0;

        if (paths.empty()) {
            std::cerr << "이미지 파일을 찾을 수 없습니다: " << path << std::endl;
            return -1;
        }

        for (std::string& p : paths) {
            cv::Mat img = cv::imread(p);

            if (img.empty()) {
                std::cerr << "읽기 실패: " << p << std::endl;
                continue;
            }

            raw_q->Push(img);
            display_q->Push(img);
            ++count;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::cout << "\n\n images num : " << count << std::endl;

        return 0;
    }

    int get_label(std::string path, LockFreeQueueSPSC<std::vector<s_label>> *labels) {
        std::vector<std::string> paths;

        cv::glob(path, paths, false);

        if (paths.empty()) {
            std::cerr << "텍스트 파일을 찾을 수 없습니다: " << std::endl;
            return -1;
        }

        int paths_size = paths.size();

        for (size_t i = 0; i < paths_size; ++i) {
            std::ifstream file(paths[i].c_str());

            std::vector<s_label> one_file;
            std::string line = "";
            s_label label;

            while (std::getline(file, line)) {
                if (line.empty()) {
                    continue;
                }
                std::stringstream split_text(line);

                if (split_text 
                    >> label.class_id
                    >> label.x1
                    >> label.y1
                    >> label.x2
                    >> label.y2
                )
                one_file.emplace_back(label);
            }

            labels->Push(one_file);
            file.close();
        }

        return paths_size;
    }

    int draw_box(LockFreeQueueSPSC<cv::Mat>* display_q, LockFreeQueueSPSC<std::vector<Detection>>* bbox_q, LockFreeQueueSPSC<cv::Mat>* final_q, bool* is_running) {
        while (is_running) {
            cv::Mat image;
            std::vector<Detection> detections;
            if (display_q->Pop(image)) {
                while (true) {
                    if(!bbox_q->Pop(detections)) {
                        continue;
                    }

                    for (const auto& det : detections) {
                        int y1 = static_cast<int>(det.y1);
                        int x1 = static_cast<int>(det.x1);
                        int y2 = static_cast<int>(det.y2); 
                        int x2 = static_cast<int>(det.x2);

                        cv::Rect rect(x1, y1, x2 - x1, y2 - y1); 
                        cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 3); 

                        std::string full_label = cv::format("%d %.2f", (int)det.class_id, det.confidence);

                        int baseline = 0;
                        cv::Size text_size = cv::getTextSize(full_label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
                        cv::rectangle(image, 
                                    cv::Point(x1, y1 - text_size.height - 5), 
                                    cv::Point(x1 + text_size.width, y1), 
                                    cv::Scalar(0, 0, 255), cv::FILLED);

                        cv::putText(image, full_label, cv::Point(x1, y1 - 5), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                        printf("%d @ (%d %d %d %d) %.3f\n", (int)det.class_id, y1, x1, y2, x2, det.confidence);
                    }
                    
                    final_q->Push(image);
                    break;
                }
            }
        }

        return 0;
    }

    float calculate_iou(const float* box1, const float* box2) {
        float x1 = std::max(box1[0], box2[0]);
        float y1 = std::max(box1[1], box2[1]);
        float x2 = std::min(box1[2], box2[2]);
        float y2 = std::min(box1[3], box2[3]);

        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        float union_area = area1 + area2 - intersection;

        return (union_area <= 0) ? 0 : intersection / union_area;
    }

    int test() {
        //std::string images_file_path = "test_folder/test/*.jpg";
        //std::string labels_file_path = "test_folder/label/test/*.txt";

        std::string images_file_path = "test_folder/102_class/*.jpg";
        std::string labels_file_path = "test_folder/102_class/label/*.txt";

        LockFreeQueueSPSC<std::vector<s_label>> labels;
        LockFreeQueueSPSC<std::vector<Detection>> detections;

        std::thread t1(get_images, images_file_path, &raw_q, &display_q);
        std::thread t2(draw_box, &display_q, &detections, &final_q, &is_running);

        int label_size = get_label(labels_file_path, &labels);
        if (label_size < 0) {
            return -1;
        }

        LaborManager lm(&is_running);
        SharedMemoryManager smm("yolo_frame", 480, 480);
        std::vector<Detection>* recive_output;
        std::vector<s_label> p_label;

        size_t count = 0;
        size_t f_count = 0;
        size_t t_count = 0;
        size_t id_f_count = 0;
        size_t id_t_count = 0;

        cv::Mat di;

        while (is_running) {
            cv::Mat image;
            if (final_q.Pop(di)) {
                std::string save_path = cv::format("try_2/single_cpp_image/%ld.jpg", count);  
                cv::imwrite(save_path, di);
            }

            if (count == label_size) {
                is_running = false;
                continue;
            }

            if (raw_q.Pop(image)) {
                smm.sendFrame(image);
            }

            recive_output = smm.receiveYoloResult();

            if (recive_output == nullptr) {
                continue;
            }

            std::vector<Detection> temp = *recive_output;
            detections.Push(temp);

            /*
            for (Detection& det : *recive_output) {
                std::cout << " - Confidence   : " << (det.confidence * 100.0) << "%" << std::endl;
                std::cout << " - Class ID     : " << (int)det.class_id << std::endl;
            }
            */

            
            labels.Pop(p_label);
            int t_o_num = p_label.size();
            int object_num = recive_output->size();

            //printf("\n%d, %d \n", object_num, t_o_num);

            ++count;

            if (t_o_num == 0) {
                if (object_num == 0) {
                    ++t_count;
                }
                else {
                    ++f_count;
                }
                continue;
            }

            /*
            if (object_num != 1 or (t_o_num == 0 and object_num > 0)) {
                ++f_count;
                printf("\nrecvie object num is %d\n", object_num);
                continue;
            }

            size_t l_id_f_count = 0;

            size_t w_count = 0;
            
            while (true) {
                if (t_o_num == l_id_f_count) {
                    ++f_count;
                    break;
                }

                Detection& det = (*recive_output)[0];
                s_label& label = p_label[w_count];

                if ((int)det.class_id == (int)label.class_id) {
                    ++t_count; 
                    break;
                }
                else {
                    ++l_id_f_count;
                }
                
                ++w_count;
            }
        }
            

        std::cout << "\n총 이미지 수 : " << count;
        std::cout << "\n정답 맞춘 수 : " << t_count;
        std::cout << "\n정답 못 맞춘 수 : " << f_count << std::endl;
        */

            if (object_num == 0) {
                continue;
            }

            for (auto& label : p_label) {
                float w = 480.0f; 
                float h = 480.0f;

                float gx = label.x1;
                float gy = label.y1;
                float gw = label.x2;
                float gh = label.y2;

                // 픽셀 단위 x1, y1, x2, y2로 덮어쓰기
                label.x1 = (gx - gw / 2.0f) * w;
                label.y1 = (gy - gh / 2.0f) * h;
                label.x2 = (gx + gw / 2.0f) * w;
                label.y2 = (gy + gh / 2.0f) * h;
            }
            /*
            if (object_num > 0) {
                Detection& det = (*recive_output)[0];
                s_label& label = p_label[0];

                printf("\n[DEBUG] Frame %zu\n", count);
                printf("PRED: Box(%.1f, %.1f, %.1f, %.1f) Class: %d\n", det.x1, det.y1, det.x2, det.y2, (int)det.class_id);
                printf("GT  : Box(%.1f, %.1f, %.1f, %.1f) Class: %d\n", label.x1, label.y1, label.x2, label.y2, (int)label.class_id);
            }
            */
            int img_tp = 0;
            std::vector<bool> matched_preds(object_num, false);

            for (const auto& gt : p_label) {
                bool found_match = false;
                float gt_box[4] = {gt.x1, gt.y1, gt.x2, gt.y2};

                for (int p_idx = 0; p_idx < object_num; ++p_idx) {
                    if (matched_preds[p_idx]) continue; 

                    Detection& det = (*recive_output)[p_idx];
                    float pred_box[4] = {det.x1, det.y1, det.x2, det.y2};

                    if ((int)gt.class_id == (int)det.class_id) {
                        float iou = calculate_iou(gt_box, pred_box); 
                        if (iou >= 0.3f) {
                            ++img_tp;
                            matched_preds[p_idx] = true; 
                            found_match = true;
                            break;
                        }
                    }
                }
            }

            if (img_tp > 0 && img_tp == t_o_num && object_num == t_o_num) {
                ++t_count; 
            } else {
                ++f_count;
            }

            id_t_count += img_tp; 
            id_f_count += (t_o_num - img_tp);
        }

        std::cout << "\n총 이미지 수 : " << count;
        std::cout << "\n정답 맞춘 수 : " << t_count << " | 객체 맞춘 수 " << id_t_count;
        std::cout << "\n정답 틀린 수 : " << f_count << " | 객체 틀린 수 " << id_f_count << std::endl;


        t1.join(); t2.join();
        smm.sendExitSignal();

        return 0;
    }

    void test_tracking_logic();
    void visualize_tracking();
    void visualize_tracking_line();

    //test를 어떻게 진행할지 잘 고민해봐야할 것 같음.
    //일단 object를 내가 만들어야겠지? 만드는거
    int tracker_test() {

        visualize_tracking_line();
        return 0;
    }
    //tracker의 현 frame 업데이트 진행.
    void tracker_update(TrackerVector* trackers) {
        for (int i = 0; i < trackers->size(); ++i) {
            Tracker& t = (*trackers)[i];
            
            t.data.past_cx += t.data.vx;
            t.data.past_cy += t.data.vy;
            
            ++t.data.missing_count; 
        }
    }

    //새로 들어온 object와 tracker가 담고 있는 object matching
    void tracker_match(Detection* object, TrackerVector* trackers) {
        float cx = (object->x1 + object->x2) * 0.5f;
        float cy = (object->y1 + object->y2) * 0.5f;

        int best_match_idx = -1;
        float min_dist_sq = 999999.0f;

        //가장 가까이에 있는 것을 찾도록 함.
        for (int i = 0; i < trackers->size(); ++i) {
            Tracker& t = (*trackers)[i];
            
            float dx = t.data.past_cx - cx;
            float dy = t.data.past_cy - cy;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq && dist_sq < MAX_DIST_SQ) {
                min_dist_sq = dist_sq;
                best_match_idx = i;
            }
        }

        if (best_match_idx != -1) {
            Tracker& matched_tr = (*trackers)[best_match_idx];
            
            matched_tr.data.vx = cx - (matched_tr.data.past_cx - matched_tr.data.vx);
            matched_tr.data.vy = cy - (matched_tr.data.past_cy - matched_tr.data.vy);
            
            matched_tr.data.past_cx = cx;
            matched_tr.data.past_cy = cy;
            

            matched_tr.data.missing_count = 0;
            matched_tr.history.add(static_cast<int>(object->class_id));

        } else {
            Tracker* new_tr = trackers->emit_back();
            
            if (new_tr != nullptr) {
                new_tr->data.past_cx = cx;
                new_tr->data.past_cy = cy;
                new_tr->data.vx = 0.0f;
                new_tr->data.vy = 0.0f;
                new_tr->data.missing_count = 0;
                new_tr->data.is_lost = false;
                new_tr->history.add(static_cast<int>(object->class_id));
            }
        }
    }

    void test_tracking_logic() {
        TrackerVector trackers;
        std::cout << "--- Tracker Simulation Test Started ---" << std::endl;

        for (int frame = 0; frame < 100; ++frame) {
            std::cout << "\n[Frame " << frame << "]" << std::endl;

            if (trackers.size() > 0) {
                tracker_update(&trackers);
            }

            std::vector<Detection> current_frame_objects;

            Detection obj1;
            float center_x1 = 100.0f + (frame * 10.0f);
            float center_y1 = 100.0f + (frame * 10.0f);
            obj1.x1 = center_x1 - 5.0f;
            obj1.x2 = center_x1 + 5.0f;
            obj1.y1 = center_y1 - 5.0f;
            obj1.y2 = center_y1 + 5.0f;
            obj1.class_id = 1;
            current_frame_objects.push_back(obj1);

            if (frame >= 2) {
                Detection obj2;
                float center_x2 = 200.0f + ((frame - 2) * 20.0f);
                float center_y2 = 50.0f;
                obj2.x1 = center_x2 - 10.0f; 
                obj2.x2 = center_x2 + 10.0f;
                obj2.y1 = center_y2 - 10.0f;
                obj2.y2 = center_y2 + 10.0f;
                obj2.class_id = 2;
                current_frame_objects.push_back(obj2);
            }

            for (int i = 0; i < current_frame_objects.size(); ++i) {
                tracker_match(&current_frame_objects[i], &trackers);
            }


            for (int i = 0; i < trackers.size(); ++i) {
                Tracker& t = trackers[i];
                std::cout << "  Tracker [" << i << "] "
                        << "| Pos: (" << t.data.past_cx << ", " << t.data.past_cy << ") "
                        << "| Vel: (" << t.data.vx << ", " << t.data.vy << ") "
                        << "| Missing: " << t.data.missing_count << "\n";
            }

            // trackers.cleanup(); // 구현하신 cleanup 호출 (missing_count 초과 시 삭제)
        }
        std::cout << "---------------------------------------" << std::endl;
    }

    #include <cmath>

    // 트래커 ID별 고유 색상 생성을 위한 함수
    cv::Scalar get_color(int id) {
        int r = (id * 123) % 255;
        int g = (id * 456) % 255;
        int b = (id * 789) % 255;
        return cv::Scalar(b, g, r); // OpenCV는 BGR 순서
    }

    void visualize_tracking() {
        TrackerVector trackers;
        cv::Mat canvas = cv::Mat::zeros(480, 480, CV_8UC3);
        
        std::cout << "--- 시각화 테스트 시작 (ESC를 누르면 종료) ---" << std::endl;

        for (int frame = 0; frame < 100; ++frame) {
            if (trackers.size() > 0) {
                tracker_update(&trackers);
            }

            std::vector<Detection> current_frame_objects;
            Detection obj1;
            float center_x1 = 50.0f + (frame * 4.0f);
            float center_y1 = 50.0f + (frame * 3.0f);
            
            obj1.x1 = center_x1 - 5.0f;
            obj1.x2 = center_x1 + 5.0f;
            obj1.y1 = center_y1 - 5.0f;
            obj1.y2 = center_y1 + 5.0f;
            obj1.class_id = 1;
            current_frame_objects.push_back(obj1);

            for (auto& obj : current_frame_objects) {
                tracker_match(&obj, &trackers);
            }

            for (int i = 0; i < trackers.size(); ++i) {
                Tracker& t = trackers[i];
                cv::Scalar color = get_color(t.data.tracker_number);
                
                // 현재 위치에 점 찍기
                cv::circle(canvas, cv::Point(t.data.past_cx, t.data.past_cy), 2, color, -1);
                
                // 현재 위치 옆에 ID 표시
                cv::putText(canvas, "ID:" + std::to_string(i), 
                            cv::Point(t.data.past_cx + 5, t.data.past_cy), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }

            // 화면 출력
            cv::imshow("Tracker Path Visualization", canvas);
            
            // 30ms 대기 (초당 약 33프레임 속도)
            if (cv::waitKey(30) == 27) break; 
        }

        cv::waitKey(0); // 종료 전 결과 유지
    }

    #include <map> // 트래커별 경로 저장을 위해 사용

    // 트래커 ID별로 좌표 리스트를 관리합니다.
    std::map<int, std::vector<cv::Point>> path_history;

    void visualize_tracking_line() {
        TrackerVector trackers;
        cv::Mat canvas = cv::Mat::zeros(480, 480, CV_8UC3);
        
        for (int frame = 0; frame < 200; ++frame) {
            // 1. 업데이트 및 가상 객체 생성 (기존 로직 동일)
            if (trackers.size() > 0) tracker_update(&trackers);

            std::vector<Detection> objects;
            Detection obj;
            float cx = 50.0f + (frame * 3.0f);
            float cy = 100.0f + std::sin(frame * 0.1f) * 50.0f;
            obj.x1 = cx-5; obj.x2 = cx+5; obj.y1 = cy-5; obj.y2 = cy+5;
            objects.push_back(obj);

            for (auto& o : objects) tracker_match(&o, &trackers);

            for (int i = 0; i < trackers.size(); ++i) {
                Tracker& t = trackers[i];
                int id = t.data.tracker_number;
                
                path_history[id].push_back(cv::Point((int)t.data.past_cx, (int)t.data.past_cy));

                const std::vector<cv::Point>& points = path_history[id];
                cv::Scalar color = get_color(id);

                for (size_t j = 1; j < points.size(); ++j) {
                    cv::line(canvas, points[j - 1], points[j], color, 2, cv::LINE_AA);
                }

                std::cout << "  Tracker [" << i << "] "
                        << "| Pos: (" << t.data.past_cx << ", " << t.data.past_cy << ") "
                        << "| Vel: (" << t.data.vx << ", " << t.data.vy << ") "
                        << "| Missing: " << t.data.missing_count << "\n";
            }

            cv::imshow("Tracking Path Line", canvas);
            if (cv::waitKey(30) >= 0) break;
        }
    }
}

int test() {
    return TestScope::tracker_test();
}

int draw_box(LockFreeQueueSPSC<cv::Mat>* display_q, LockFreeQueueSPSC<std::vector<Detection>>* bbox_q, LockFreeQueueSPSC<cv::Mat>* final_q, bool* is_running) {
    while (is_running) {
        cv::Mat image;
        std::vector<Detection> detections;
        if (display_q->Pop(image)) {
            while (true) {
                if(!bbox_q->Pop(detections)) {
                    continue;
                }

                for (const auto& det : detections) {
                    int y1 = static_cast<int>(det.y1);
                    int x1 = static_cast<int>(det.x1);
                    int y2 = static_cast<int>(det.y2); 
                    int x2 = static_cast<int>(det.x2);

                    cv::Rect rect(x1, y1, x2 - x1, y2 - y1); 
                    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 3); 

                    std::string full_label = cv::format("%d %.2f", (int)det.class_id, det.confidence);

                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(full_label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
                    cv::rectangle(image, 
                                cv::Point(x1, y1 - text_size.height - 5), 
                                cv::Point(x1 + text_size.width, y1), 
                                cv::Scalar(0, 0, 255), cv::FILLED);

                    cv::putText(image, full_label, cv::Point(x1, y1 - 5), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                    printf("%d @ (%d %d %d %d) %.3f\n", (int)det.class_id, y1, x1, y2, x2, det.confidence);
                }
                
                final_q->Push(image);
                break;
            }
        }
    }

    return 0;
}

int run() {
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    MotionDetector detector;
    TrackerVector tracker_vector;
    LaborManager lm(&is_running);
    SharedMemoryManager smm("yolo_frame", 480, 480);
    std::vector<Detection>* recive_output;
    LockFreeQueueSPSC<std::vector<Detection>> detections;

    std::thread t1([&]() {
        lm.capture_worker(pipe, raw_q, display_q);
    });
    std::thread t2([&]() {
        lm.diff_worker(detector, raw_q, motion_q);
    });
    std::thread t3([&]() {
        lm.mask_worker(motion_q, rect_q, mask_q);
    });
    std::thread t4([&]() {
        lm.rect_worker(mask_q, rect_q);
    });
    std::thread t5([&]() {
        lm.new_crop_worker(rect_q, display_q, filtered_frame_q);
    });

    std::thread t6(draw_box, &display_q, &detections, &final_q, &is_running);
    //std::thread t5([&]() {
    //    lm.crop_worker(rect_q, display_q, chips_q);
    //});
    //std::thread t5([&]() {
    //    lm.draw_worker(rect_q, display_q, final_q);
    //});
    //std::thread t6([&]() {
    //    recive_output = smm.receiveYoloResult();
    //});

    size_t count = 0; 
    while (is_running) {
        cv::Mat display_frame;
        cv::Mat display;
        std::vector<cv::Mat> frames;
        if (!final_q.Pop(display)) {
            if (chips_q.Pop(frames)) {
                for (cv::Mat& frame : frames) {
                    smm.sendFrame(frame);
                }
                //printf("frame count is {%zu}\n", frames.size());
            }
        }
        else {
            std::string save_path = cv::format("try_3/image/%ld.jpg", count);  
            cv::imwrite(save_path, display);
            cv::imshow("boxes", display);
            if (cv::waitKey(1) == 'q') is_running = false;
        }

        if (filtered_frame_q.Pop(display_frame)) {
            smm.sendFrame(display_frame);
        }

        recive_output = smm.receiveYoloResult();
        
        if (recive_output == nullptr) {
            continue;
        }
        std::vector<Detection> temp = *recive_output;
        detections.Push(temp);

        if (!recive_output->empty()) {
            for (Detection& det : *recive_output) {
                std::cout << " - Confidence   : " << (det.confidence * 100.0) << "%" << std::endl;
                std::cout << " - Class ID     : " << (int)det.class_id << std::endl;
            }
        }
        ++count;
    }

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); //t6.join();
    return 0;
}

int main() { 
    return run(); 
    //return test();
}