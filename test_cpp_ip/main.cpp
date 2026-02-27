#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include <stdlib.h>


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
        float x, y, x1, y1;
    };

    int get_images(std::string path, LockFreeQueueSPSC<cv::Mat>* raw_q) {
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
            ++count;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
                    >> label.x
                    >> label.y
                    >> label.x1
                    >> label.y1
                )
                one_file.emplace_back(label);
            }

            labels->Push(one_file);
            file.close();
        }

        return paths_size;
    }

    int test() {
        std::string images_file_path = "test_folder/test/*.jpg";
        std::string labels_file_path = "test_folder/label/test/*.txt";

        std::thread t1(get_images, images_file_path, &raw_q);

        LockFreeQueueSPSC<std::vector<s_label>> labels;

        int label_size = get_label(labels_file_path, &labels);
        if (label_size == -1) {
            return -1;
        }

        LaborManager lm(&is_running);
        SharedMemoryManager smm("yolo_frame", 480, 480);
        std::vector<Detection>* recive_output;
        std::vector<s_label> p_label;

        size_t count = 0;
        size_t f_count = 0;
        size_t t_count = 0;
        size_t object_count = 0;
        size_t id_f_count = 0;
        size_t id_t_count = 0; 

        while (is_running) {
            cv::Mat image;

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

            labels.Pop(p_label);
            int t_o_num = p_label.size();
            int object_num = recive_output->size();

            //printf("\n%d, %d \n", object_num, t_o_num);

            ++count;
            object_count += t_o_num;

            if (object_num != t_o_num) {
                ++f_count;
                id_f_count += t_o_num;
                continue;
            }

            size_t l_id_f_count = 0;
            size_t l_id_t_count = 0;  

            size_t w_count = 0;
            
            while (true) {
                if (object_num == w_count) {
                    break;
                }

                Detection& det = (*recive_output)[w_count];
                s_label& label = p_label[w_count];

                if (det.class_id != label.class_id) {
                    ++l_id_f_count;
                }
                else {
                    ++l_id_t_count;
                }
                
                ++w_count;
            }

            std::cout << "\n\n" << std::string(10, '-') << std::endl;

            std::cout << "\n정답 갯수 : " << t_o_num;
            std::cout << "\n맞춘 갯수 : " << l_id_t_count;
            std::cout << "\n틀린 갯수 : " << l_id_f_count;

            id_t_count += l_id_t_count;
            id_f_count += l_id_f_count;

            if (l_id_t_count == t_o_num) {
                ++t_count;  
            }
            else {
                ++f_count;
            }
        }

        std::cout << "\n총 이미지 수 : " << count;
        std::cout << "\n정답 맞춘 수 : " << t_count;
        std::cout << "\n정답 못 맞춘 수 : " << f_count << std::endl;

        std::cout << "\n총 객체 수 : " << object_count;
        std::cout << "\n정답 맞춘 수 : " << id_t_count;
        std::cout << "\n정답 못 맞춘 수 : " << id_f_count << std::endl;

        t1.join();
        smm.sendExitSignal();

        return 0;
    }
}

int test() {
    return TestScope::test();
}

int run() {
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    MotionDetector detector;
    LaborManager lm(&is_running);
    SharedMemoryManager smm("yolo_frame", 480, 480);
    std::vector<Detection>* recive_output;

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
    //std::thread t5([&]() {
    //    lm.crop_worker(rect_q, display_q, chips_q);
    //});
    //std::thread t5([&]() {
    //    lm.draw_worker(rect_q, display_q, final_q);
    //});
    //std::thread t6([&]() {
    //    recive_output = smm.receiveYoloResult();
    //});

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
        }

        if (filtered_frame_q.Pop(display_frame)) {
            smm.sendFrame(display_frame);
            //cv::imshow("boxes", display_frame);
            //if (cv::waitKey(1) == 'q') is_running = false;
        }

        recive_output = smm.receiveYoloResult();

        if (!recive_output->empty()) {
            for (Detection& det : *recive_output) {
                std::cout << " - Confidence   : " << (det.confidence * 100.0) << "%" << std::endl;
                std::cout << " - Class ID     : " << (int)det.class_id << std::endl;
            }
        }
    }

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); //t6.join();
    return 0;
}

int main() { 
    //return run(); 
    return test();
}