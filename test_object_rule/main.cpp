#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>

#include <stdlib.h>

#include "define.h"

#include "MotionDetector.hpp"
#include "LFQSPSC.h"
#include "SharedMemoryManager.hpp"
#include "LaborManager.hpp"

std::string OBJECT_LOCATE_LIST_DIR = "../test_object_/scenario_zero_test_999";

const int REPEAT = 1000;   

int option = 1;

size_t miss_count = 0;
size_t not_move_count = 0;

/*
std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        tokens.push_back(s.substr(last, next - last));
        last = next + delimiter.length();
    }
    tokens.push_back(s.substr(last));
    return tokens;
}

bool parseFrameLine(const std::string& line, 
                    std::vector<cv::Rect>& target_regions, 
                    std::vector<Detection>& detections) {
    
    target_regions.clear();
    detections.clear();

    auto main_parts = split(line, " | ");
    if (main_parts.size() < 2) return false;

    std::stringstream ss_header(main_parts[0]);
    std::string dummy, frame_val, motion_label, first_val;

    ss_header >> dummy >> frame_val >> motion_label >> first_val;

    if (first_val != "None") {
        try {
            float mx1 = std::stof(first_val);
            float my1, mx2, my2;
            if (ss_header >> my1 >> mx2 >> my2) {
                target_regions.emplace_back(cv::Rect(cv::Point(mx1, my1), cv::Point(mx2, my2)));
            }
        } catch (...) {

        }
    }
    else {

    }

    auto obj_list = split(main_parts[1], " / ");
    for (const auto& obj_str : obj_list) {
        std::stringstream ss_obj(obj_str);
        Detection det;
        if (ss_obj >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.confidence >> det.class_id) {
            detections.push_back(det);
        }
    }

    return !detections.empty();
}

bool get_object_locate(LockFreeQueueSPSC<std::vector<Detection>>* object_pos_list, std::string* path, LockFreeQueueSPSC<std::vector<cv::Rect>>* mask_q) {
    if (!path || !object_pos_list) return false;

    std::ifstream file(*path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    bool success = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        for (char &c : line) {
            if (c == ',') c = ' ';
        }

        std::stringstream ss(line);
        Detection det;
        std::vector<Detection> dets;

        if (option == 1) {
            std::vector<cv::Rect> rects;

            success = parseFrameLine(line, rects, dets);

            object_pos_list->Push(dets);
            mask_q->Push(rects);
            continue;
        }

        if (ss >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.confidence >> det.class_id) {
            dets.emplace_back(det); 
            success = true;
        }
        object_pos_list->Push(dets);
    }

    file.close();
    return success;
}

void save_tracker_anomaly(std::string* file_path, TrackerVector& trackers, bool* is_running) {
    bool has_anomaly = false;
    std::string scenario_name = *file_path;

    while(*is_running) {
        if (trackers.size() > 1) {
            has_anomaly = true;
        }

        if (has_anomaly) {
            std::ofstream log_file("tracker_anomaly_log.txt", std::ios::app); 
            if (log_file.is_open()) {
                log_file << "----------------------------------------------------" << std::endl;
                log_file << "[Anomaly Detected]" << std::endl;
                log_file << "File: " << scenario_name << std::endl;
                log_file << "Active Trackers: " << trackers.size() << std::endl;
                
                for (int i = 0; i < trackers.size(); ++i) {
                    Tracker& t = trackers[i];
                    log_file << "  - Tracker Index [" << i << "] ID: " << t.data.tracker_number 
                            << " | Pos: (" << t.data.past_cx << ", " << t.data.past_cy << ")"
                            << " | Miss: " << t.data.missing_count << std::endl;
                }
                log_file << "----------------------------------------------------" << std::endl << std::endl;
                log_file.close();
            }
            has_anomaly = false;
        }
    }
}

void display_tracker_monitor(TrackerVector& trackers, std::string* file_path) {
    bool is_print = true;

    is_print = false;
 
    if (is_print) {
        std::cout << "\033[2J\033[1;1H";
        std::cout << "now ... >> " << *file_path << "<< ing...";
        std::cout << "\n================================ [ Tracker Monitor ] ================================" << std::endl;
        std::cout << " Active Trackers: " << trackers.size() << " / " << trackers.capacity() << " | Zone: [" << X_MIN << " ~ " << X_MAX << "]" << std::endl;
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;
        std::cout << "  ID  |  Status  | InZone |    Center(X,Y)    |    Velocity   | Miss | Class | AvgSize " << std::endl;
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;

        for (int i = 0; i < trackers.size(); ++i) {
            Tracker& t = trackers[i];
            
            std::string status = (t.data.missing_count > 0) ? "MISSING" : "TRACKED";
            
            std::string zone_status = check_x_is_here((int)t.data.past_cx) ? "  IN  " : " OUT  ";

            std::cout << " " << std::setw(4) << t.data.tracker_number << " | "
                    << std::setw(8) << status << " | "
                    << zone_status << " | " 
                    << std::fixed << std::setprecision(1) 
                    << std::setw(7) << t.data.past_cx << "," << std::setw(7) << t.data.past_cy << " | "
                    << "v:" << std::setw(4) << t.data.vx << "," << std::setw(4) << t.data.vy << " | "
                    << std::setw(4) << t.data.missing_count << " | ";

            if (t.history.get_infer_class() != nullptr) {
                std::cout << std::setw(5) << *(t.history.get_infer_class()) << " | ";
            } else {
                std::cout << std::setw(5) << "N/A" << " | ";
            }

            if (t.history.get_bbox_average() != nullptr) {
                std::cout << std::setw(7) << (int)*(t.history.get_bbox_average()) << std::endl;
            } else {
                std::cout << std::setw(7) << "N/A" << std::endl;
            }
        }
        std::cout << "=====================================================================================" << std::endl;
    }
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
int count = 0;

extern std::map<int, std::vector<cv::Point>> path_history;

int run(LockFreeQueueSPSC<std::vector<Detection>>* object_pos, LockFreeQueueSPSC<std::vector<cv::Rect>>* mask_q, std::string* file_path) {
    path_history.clear();
    std::vector<Detection> objects;
    std::vector<cv::Rect> rects;
    LockFreeQueueSPSC<std::vector<Detection>> object_q;
    TrackerVector trackers;
    cv::Mat canvas = cv::Mat::zeros(480, 480, CV_8UC3);
    ++count;

    std::string output_video = "video_1/tracker_record " + std::to_string(count) + ".mp4";
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // 코덱 설정
    double fps = 20.0;
    cv::Size frame_size(480, 480);

    cv::VideoWriter writer(output_video, fourcc, fps, frame_size);

    if (!writer.isOpened()) {
        std::cerr << "동영상 파일을 열 수 없습니다!" << std::endl;
    }

    bool is_running = true;

    LaborManager lm(&is_running);
    std::thread t1([&]() {
        lm.track_worker(object_q, &trackers);
    });

    std::thread t2([&]() {
        save_tracker_anomaly(file_path, trackers, &is_running);
    });
    bool one_time = false;
    while (true) {
        if (!object_pos->Pop(objects)) {
            is_running = false;
            break;
        }
        if (!mask_q->Pop(rects)) {
            is_running = false;
            break;
        }
        int size = objects.size();
        keepBestDetectionByCenter(objects, rects);
        size -= objects.size();

        object_q.Push(objects);

        display_tracker_monitor(trackers, file_path);
        
        if (option == 1) {
            if (rects.empty()) {
                one_time = true;
            }
            canvas = cv::Mat::zeros(480, 480, CV_8UC3);
            mut_draw_tracker_visualization(canvas, trackers, &size, &path_history);
        }
        else{
            draw_tracker_visualization(canvas, trackers);
        }            
        if (writer.isOpened()) {
            writer.write(canvas);
        }
        //cv::imshow("Tracker Monitor", canvas);
        //cv::waitKey(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    t1.join(); t2.join();
    miss_count += *trackers.get_how_many_erase();
    if (one_time) {
        ++not_move_count;
    }
    return 0;
}

void object_test() {
    LockFreeQueueSPSC<std::vector<Detection>> objects_locate;
    LockFreeQueueSPSC<std::vector<cv::Rect>> mask_q;
    bool is_done = false;

    for (int i = 1; i < REPEAT; ++i) {
        std::ostringstream oss;
        
        oss << "scenario_" << std::setfill('0') << std::setw(3) << i << ".txt";
        
        std::string file_path = OBJECT_LOCATE_LIST_DIR + "/" + oss.str();
        
        std::cout << "Reading: " << file_path << "..." << std::endl;

        if (get_object_locate(&objects_locate, &file_path, &mask_q)) {
            std::cout << "Successfully pushed data from " << file_path << std::endl;
        } 
        else {
            std::cerr << "Failed to read or empty file: " << file_path << std::endl;
            continue;
        }

        run(&objects_locate, &mask_q, &file_path);
    }
}


int main() {
    object_test();
    std::cout << "miss_count = " << miss_count << std::endl;
    std::cout << "not_move_count = " << not_move_count << std::endl;
    return 0;
}
*/