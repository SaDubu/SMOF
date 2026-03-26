#include "define.h"

std::string record_dir = "./record";
LockFreeQueueSPSC<std::string> file_list;

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

float calculate_iou(float x1_a, float y1_a, float x2_a, float y2_a, int area_a,
                    float x1_b, float y1_b, float x2_b, float y2_b, int area_b) {
    float x_left   = std::max(x1_a, x1_b);
    float y_top    = std::max(y1_a, y1_b);
    float x_right  = std::min(x2_a, x2_b);
    float y_bottom = std::min(y2_a, y2_b);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    float result = intersection_area / (area_a + area_b - intersection_area); 

    return result;
}

float calculate_2Box_size_ratio(int area_a, int area_b) {
    float result = std::min(area_a, area_b) / std::max(area_a, area_b);

    return result;
}

bool check_x_is_here(int object_cx) {
    bool result = false;

    if (object_cx >= X_MIN && object_cx <= X_MAX) {
        return true;
    }
    return false;
}

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

void keepBestDetectionByCenter(std::vector<Detection>& detections, std::vector<cv::Rect>& target_regions) {
    bool change = false;
    if (detections.size() == 1) {
        Detection det = detections[0];
        if (det.x1 == det.x2 == det.y1 == det.y2) {
            change = true;
        }
    }

    std::vector<Detection> result;
    std::vector<bool> is_selected(detections.size(), false);

    for (int x = 0; x < target_regions.size(); ++x) {
        cv::Rect& target_region = target_regions[x];
        float target_cx = target_region.x + target_region.width * 0.5f;
        float target_cy = target_region.y + target_region.height * 0.5f;

        int best_idx = -1;
        float min_dist_sq = MAX_DIST_SQ;
        float best_score = -1.0f;

        for (int i = 0; i < detections.size(); ++i) {
            Detection& det = detections[i];
            if (change) {
                det.class_id = -2.0f;
                det.x1 = target_region.x;
                det.x2 = target_region.x + target_region.width;
                det.y1 = target_region.y;
                det.y2 = target_region.y + target_region.height;
            }
            
            float cx = det.x1 + (det.x2 - det.x1) * 0.5f;
            float cy = det.y1 + (det.y2 - det.y1) * 0.5f;

            float dx = cx - target_cx;
            float dy = cy - target_cy;
            
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq - EPSILON) {
                min_dist_sq = dist_sq;
                best_score = det.confidence;
                best_idx = i;
            } 

            else if (std::abs(dist_sq - min_dist_sq) <= EPSILON) {
                if (det.confidence > best_score) {
                    min_dist_sq = dist_sq;
                    best_score = det.confidence;
                    best_idx = i;
                }
            }
        }

        if (best_idx != -1) {
            result.emplace_back(detections[best_idx]);
            is_selected[best_idx] = true;
        }
    }

    detections.clear();
    detections = std::move(result);
}

void mut_draw_tracker_visualization(cv::Mat& frame, TrackerVector& trackers, int* rejected_detections_num, std::map<int, std::vector<cv::Point>>* path_history) {
    int margin_x = 10;
    int margin_y = 25;
    cv::Scalar text_color(255, 255, 255);

    std::string tracking_ids = "Tracking Class IDs: ";
    if (trackers.size() == 0) {
        tracking_ids += "None";
    } else {
        for (size_t i = 0; i < trackers.size(); ++i) {
            tracking_ids += std::to_string(*(trackers[i].history.get_infer_class()));
            if (i < trackers.size() - 1) tracking_ids += ", ";
        }
    }

    std::string rejected_count_str = "Rejected Objs: ";
    if (rejected_detections_num != nullptr) {
        rejected_count_str += std::to_string(*rejected_detections_num);
    } else {
        rejected_count_str += "0";
    }
    
    cv::putText(frame, rejected_count_str, cv::Point(margin_x, margin_y + 25), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);

    cv::putText(frame, tracking_ids, cv::Point(margin_x, margin_y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv::LINE_AA);
    
    cv::Scalar zone_color(100, 100, 100); 
    cv::line(frame, cv::Point(X_MIN, 0), cv::Point(X_MIN, frame.rows), zone_color, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(X_MAX, 0), cv::Point(X_MAX, frame.rows), zone_color, 1, cv::LINE_AA);

    for (int i = 0; i < trackers.size(); ++i) {
        TrackerData& d = trackers[i].data;
        int id = d.tracker_number;
        cv::Scalar unique_color((id * 77) % 255, (id * 135) % 255, (id * 213) % 255);

        std::vector<cv::Point>& current_path = (*path_history)[id]; 

        if (d.missing_count == 0) {
            current_path.push_back(cv::Point((int)d.past_cx, (int)d.past_cy));
            
            if (current_path.size() > 60) {
                current_path.erase(current_path.begin());
            }
        }
        for (size_t j = 1; j < current_path.size(); ++j) {
            cv::line(frame, current_path[j - 1], current_path[j], unique_color, 1, cv::LINE_AA);
        }

        cv::Rect rect(cv::Point(d.past_x1, d.past_y1), cv::Point(d.past_x2, d.past_y2));
        cv::Scalar box_color = (d.missing_count > 0) ? cv::Scalar(0, 0, 255) : unique_color;
        int thickness = (d.missing_count > 0) ? 2 : 1; // 미스 시 더 두껍게 강조

        cv::rectangle(frame, rect, box_color, thickness);

        std::string label = "ID: " + std::to_string(id);
        if (d.missing_count > 0) label += " (Miss)";
        
        cv::putText(frame, label, cv::Point(d.past_x1, d.past_y1 - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv::LINE_AA);

        cv::Point center(d.past_cx, d.past_cy);
        cv::Point velocity_tip(d.past_cx + d.vx * 5, d.past_cy + d.vy * 5);
        
        cv::circle(frame, center, 2, box_color, -1);
        cv::arrowedLine(frame, center, velocity_tip, box_color, 2, 8, 0, 0.3);
    }
}

void move_frame_save(cv::Mat& frame, std::vector<cv::Rect>& move_area) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss_name;
    ss_name << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") 
            << "_" << std::setfill('0') << std::setw(3) << ms.count();
    
    std::string base_filename = ss_name.str();
    file_list.Push(base_filename);

    std::string img_path = record_dir + "/" + base_filename + ".jpeg";
    cv::imwrite(img_path, frame, {cv::IMWRITE_JPEG_QUALITY, 100});

    std::string data_path = record_dir + "/" + base_filename + ".txt";
    std::ofstream dataFile(data_path);

    if (dataFile.is_open()) {
        dataFile << "MOTION: ";
        for (size_t i = 0; i < move_area.size(); ++i) {
            const auto& r = move_area[i];
            dataFile << std::fixed << std::setprecision(4) 
                     << (float)r.x << " " << (float)r.y << " " 
                     << (float)(r.x + r.width) << " " << (float)(r.y + r.height);

            if (i < move_area.size() - 1) {
                dataFile << " / ";
            }
        }
        dataFile << "\n";
        dataFile.close();
    }
}

std::vector<cv::Rect> get_target_regions_from_file(std::string& filepath) {
    std::string abs_path = record_dir + "/" + filepath;
    std::vector<cv::Rect> target_regions;
    std::ifstream dataFile(abs_path);
    
    if (!dataFile.is_open()) {
        std::cerr << "Error: Could not open file " << abs_path << std::endl;
        return target_regions;
    }

    std::string line;
    if (std::getline(dataFile, line)) {
        size_t start_pos = line.find("MOTION: ");
        if (start_pos == std::string::npos) return target_regions;

        std::string data = line.substr(start_pos + 8); 
        std::stringstream ss(data);
        std::string segment;

        while (std::getline(ss, segment, '/')) {
            std::stringstream seg_ss(segment);
            float mx1, my1, mx2, my2;

            if (seg_ss >> mx1 >> my1 >> mx2 >> my2) {
                target_regions.emplace_back(
                    cv::Point(static_cast<int>(mx1), static_cast<int>(my1)),
                    cv::Point(static_cast<int>(mx2), static_cast<int>(my2))
                );
            }
        }
    }

    dataFile.close();
    delete_target_file(abs_path);
    return target_regions;
}

cv::Mat get_target_image_from_file(std::string& filepath) {
    std::string abs_path = record_dir + "/" + filepath;
    cv::Mat frame;

    frame = cv::imread(abs_path);
    delete_target_file(abs_path);
    return frame;
}

LockFreeQueueSPSC<std::string>* get_file_list() {
    LockFreeQueueSPSC<std::string>* result = new LockFreeQueueSPSC<std::string>(); 
    int count = 0;
    std::string file_path = "";
    while (file_list.Pop(file_path)) { 
        result->Push(file_path);
        ++count;
    }
    if (count == 0) {
        delete result;
        result = nullptr;
    }
    return result;
}

void delete_target_file(std::string& filepath) {
    std::remove(filepath.c_str());
}


void display_tracker_monitor(TrackerVector& trackers) {
    bool is_print = true;
 
    if (is_print) {
        std::cout << "\033[2J\033[1;1H";
        std::cout << "\n================================ [ Tracker Monitor ] ================================" << std::endl;
        std::cout << " Active Trackers: " << trackers.size() << " / " << trackers.capacity() << " | Zone: [" << X_MIN << " ~ " << X_MAX << "]" << std::endl;
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;
        std::cout << "  ID  |  Status  | InZone |    Center(X,Y)    |    Velocity   | Miss | Class | AvgSize " << std::endl;
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;

        for (int i = 0; i < trackers.size(); ++i) {
            Tracker& t = trackers[i];
            
            std::string status = (t.data.missing_count > 0) ? "MISSING" : "TRACKED";
            
            std::string zone_status = t.data.checked_in ? "  IN  " : " OUT  ";

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