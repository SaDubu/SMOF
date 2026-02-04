#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>


#include "MotionDetector.hpp"

std::vector<std::string> CamTest() {
    std::vector<std::string> working_indices;
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

    std::cout << "--- 비디오 장치 스캔 시작 ---" << std::endl;

    int total_nodes = sizeof(video_nodes) / sizeof(video_nodes[0]);
    //pipi format
    std::string pipe = ""; 
    //"v4l2src device=" + video_nodes[number] + " ! video/x-raw,format=NV12,width=640,height=480 ! videoconvert ! appsink";

    for (int i = 0; i < total_nodes; ++i) {
        pipe = "v4l2src device=" + video_nodes[i] + " ! video/x-raw,format=NV12,width=480,height=480 ! videoconvert ! appsink";
        // CAP_V4L2를 명시적으로 사용하여 시도
        cv::VideoCapture cap;

        cap.open(pipe, cv::CAP_GSTREAMER);
        
        if (cap.isOpened()) {
            cv::Mat frame;
            // 3. 실제 프레임 획득 시도 (가짜 노드면 여기서 실패함)
            cap >> frame; 
            
            if (!frame.empty()) {
                std::cout << "[SUCCESS] 카메라 발견: " << video_nodes[i] << std::endl;
                working_indices.emplace_back(pipe);
                cap.release();
                break; // 찾았으니 중단
            }
        }
        cap.release(); // 열렸으나 비어있거나, 안 열린 경우 해제
        std::cout << "[SKIP] " << video_nodes[i] << " 는 카메라가 아님." << std::endl;
    }

    std::cout << "--- 스캔 완료 ---" << std::endl;

    if (working_indices.empty()) {
        std::cout << "사용 가능한 카메라 장치를 찾지 못했습니다." << std::endl;
    } else {
        std::cout << "발견된 장치 인덱스: ";
        for (std::string idx : working_indices) std::cout << idx << " ";
        std::cout << std::endl;
    }

    return working_indices;
}

std::queue<cv::Mat> frame_queue;
std::mutex mtx;
std::condition_variable cv_cond;
bool is_running = true;
const int MAX_QUEUE_SIZE = 5;

void capture_worker(std::string pipe) {
    cv::VideoCapture cap;
    cap.open(pipe, cv::CAP_GSTREAMER);
    
    while (is_running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(mtx);
            // 큐가 너무 꽉 차면 가장 오래된 프레임을 버리고 최신걸 유지 (선택 사항)
            if (frame_queue.size() >= MAX_QUEUE_SIZE) {
                frame_queue.pop(); 
            }
            frame_queue.push(frame.clone());
        }
        cv_cond.notify_one(); // 데이터를 넣었으니 처리 스레드 깨우기
    }
}

int main() {  
    std::vector<std::string> a = CamTest();
    std::string pipe = a[0];
    MotionDetector detector;
    cv::Mat frame, motionLog;

    std::thread t1(capture_worker, pipe);

    while (is_running) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            // 큐에 데이터가 올 때까지 대기 (CPU 점유율 감소)
            cv_cond.wait(lock, [] { return !frame_queue.empty() || !is_running; });

            if (!is_running) break;

            frame = frame_queue.front();
            frame_queue.pop();
        }

        motionLog = detector.process(frame);

        cv::imshow("Motion Result", motionLog);
        if (cv::waitKey(1) == 'q') {
            is_running = false;
            cv_cond.notify_all();
            break;
        }
    }

    t1.join();
    return 0;
}