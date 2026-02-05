#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <stdlib.h>

#include "MotionDetector.hpp"


std::queue<cv::Mat> raw_q, motion_q, final_q;
std::mutex m1, m2, m3;
std::condition_variable cv1, cv2, cv3;
bool is_running = true;
const int MAX_QUEUE_SIZE = 15;

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

cv::Mat mask_moving_area(cv::Mat motionImage) {
    cv::Mat binary, morph;

    cv::threshold(motionImage, binary, 20, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
    cv::dilate(binary, morph, kernel);
    cv::erode(morph, morph, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat result = cv::Mat::zeros(morph.size(), CV_8UC1);
    cv::drawContours(result, contours, -1, cv::Scalar(255), -1);

    return result;
}

void capture_worker(std::string pipe) {
    cv::VideoCapture cap;
    cap.open(pipe, cv::CAP_GSTREAMER);
    
    while (is_running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(m1);
            if (raw_q.size() >= MAX_QUEUE_SIZE) {
                raw_q.pop(); 
            }
            raw_q.push(frame.clone());
        }
        cv1.notify_one();
    }
}

void diff_worker(MotionDetector detector) {
    while(is_running) {
        cv::Mat frame, motionLog;
        {
            std::unique_lock<std::mutex> lock(m1);
            cv1.wait(lock, [] { return !raw_q.empty() || !is_running; });

            if (!is_running) break;

            frame = raw_q.front();
            raw_q.pop();
        }

        motionLog = detector.process(frame);
        {
            std::lock_guard<std::mutex> lock(m2);
            if (motion_q.size() >= MAX_QUEUE_SIZE) {
                motion_q.pop();
            }
            motion_q.push(motionLog);
        }
        cv2.notify_one();
    }
}

void mask_worker() {
    while (is_running) {
        cv::Mat local_motion;
        {
            std::unique_lock<std::mutex> lock(m2);
            cv2.wait(lock, [] { return !motion_q.empty() || !is_running; });
            if (!is_running) break;

            local_motion = motion_q.front();
            motion_q.pop();
        }

        cv::Mat result = mask_moving_area(local_motion);

        {
            std::lock_guard<std::mutex> lock(m3);
            if (final_q.size() >= MAX_QUEUE_SIZE) final_q.pop();
            final_q.push(result);
        }
        cv3.notify_one();
    }
}

int main() {  
    std::string pipe = CamTest();
    if (pipe.empty()) {
        printf("out\n");
        return -1;
    }
    MotionDetector detector;
    cv::Mat frame, motionLog, result;

    std::thread t1(capture_worker, pipe);
    std::thread t2(diff_worker, detector);
    std::thread t3(mask_worker);

    while (is_running) {
        cv::Mat display_frame;
        {
            std::unique_lock<std::mutex> lock(m3);
            cv3.wait(lock, [] { return !final_q.empty() || !is_running; });
            if (!is_running) break;

            display_frame = final_q.front();
            final_q.pop();
        }

        cv::imshow("Final Inkjet Mask", display_frame);
        if (cv::waitKey(1) == 'q') is_running = false;
    }

    cv1.notify_all(); cv2.notify_all(); cv3.notify_all();
    t1.join(); t2.join(); t3.join();
    return 0;
}