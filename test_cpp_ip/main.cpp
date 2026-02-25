#include <opencv2/opencv.hpp>

#include <iostream>
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
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cout << "images num : " << count << std::endl;

    return 0;
}

int test() {
    std::string images_file_path = "test_folder/test/*.jpg";

    std::thread t1(get_images, images_file_path, &raw_q);

    LaborManager lm(&is_running);
    SharedMemoryManager smm("yolo_frame", 480, 480);
    std::vector<Detection> recive_output;

    while (is_running) {
        cv::Mat image;

        if (raw_q.Pop(image)) {
            smm.sendFrame(image);
        }

        recive_output = smm.receiveYoloResult();

        if (!recive_output.empty()) {
            for (Detection& det : recive_output) {
                std::cout << " - Confidence   : " << (det.confidence) << std::endl;
                std::cout << " - Class ID     : " << (int)det.class_id << std::endl;
            }
        }
    }

    t1.join();

    return 0;
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
    std::vector<Detection> recive_output;

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

        if (!recive_output.empty()) {
            for (Detection& det : recive_output) {
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