/*
#include <iostream>
#include <fcntl.h>      // For O_* constants
#include <sys/mman.h>   // For shared memory
#include <unistd.h>     // For ftruncate
#include <semaphore.h>  // For semaphores
#include <opencv2/opencv.hpp>

// 설정값 정의
const char* SHM_NAME = "/yolo_frame_shm";
const char* SEM_FULL_NAME = "/yolo_sem_full";   // 데이터가 찼음을 알림
const char* SEM_EMPTY_NAME = "/yolo_sem_empty"; // 데이터가 비었음을 알림

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int CHANNELS = 3; // RGB
const int DATA_SIZE = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS;

int main() {
    // 1. 카메라 설정 (OpenCV)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: 카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    // 카메라 해상도 강제 설정 (데이터 크기 일치를 위해)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    // 2. Shared Memory 생성 및 열기
    // O_CREAT: 없으면 생성, O_RDWR: 읽기/쓰기 모드
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Shared memory failed" << std::endl;
        return -1;
    }

    // 메모리 크기 설정 (이 과정이 없으면 Bus Error 발생 가능)
    ftruncate(shm_fd, DATA_SIZE);

    // 메모리 매핑 (Process의 메모리 주소공간에 연결)
    unsigned char* shm_ptr = (unsigned char*)mmap(0, DATA_SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // 3. Semaphore 설정 (동기화)
    // 기존에 존재할 수 있으니 unlink 후 생성
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_EMPTY_NAME);

    // sem_empty는 1로 초기화 (처음엔 쓸 수 있음)
    // sem_full은 0으로 초기화 (처음엔 읽을 데이터 없음)
    sem_t* sem_full = sem_open(SEM_FULL_NAME, O_CREAT, 0666, 0);
    sem_t* sem_empty = sem_open(SEM_EMPTY_NAME, O_CREAT, 0666, 1);

    std::cout << ">>> C++ Producer Started. Press Ctrl+C to stop." << std::endl;

    cv::Mat frame, rgb_frame;

    while (true) {
        // [Wait] 버퍼가 비워질 때까지 대기 (Python이 읽어갈 때까지 멈춤)
        // 만약 Python이 느리면 C++도 여기서 같이 느려짐 (프레임 동기화)
        sem_wait(sem_empty); 

        // 카메라 읽기
        cap >> frame;
        if (frame.empty()) break;

        // OpenCV는 기본이 BGR이므로, YOLO(RGB)를 위해 색상 변환
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // [Critical Section] 공유 메모리에 데이터 복사
        // memcpy(목적지, 소스, 크기) -> 매우 빠름
        memcpy(shm_ptr, rgb_frame.data, DATA_SIZE);

        // [Signal] 데이터가 준비되었다고 알림 (Python이 깨어남)
        sem_post(sem_full);
        
        // (선택사항) 모니터링용 화면 출력 (성능 저하 원인이 될 수 있음)
        // cv::imshow("C++ Sender", frame);
        // if (cv::waitKey(1) == 27) break;
    }

    // 4. 리소스 정리 (중요)
    munmap(shm_ptr, DATA_SIZE);
    close(shm_fd);
    shm_unlink(SHM_NAME);
    sem_close(sem_full);
    sem_close(sem_empty);
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_EMPTY_NAME);

    return 0;
}
    */

// test 할 때는 cam 사용 하지 않고 image를 가지고 진행함.
//위 코드가 실제 cam을 가지고 진행하는 것
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // sort
#include <filesystem>   // C++17 파일시스템
#include <thread>       // sleep
#include <chrono>       // time

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <opencv4/opencv2/opencv.hpp>

namespace fs = std::filesystem;

// 설정값 (Python과 일치해야 함)
const char* SHM_NAME = "/yolo_frame_shm";
const char* SEM_FULL_NAME = "/yolo_sem_full";
const char* SEM_EMPTY_NAME = "/yolo_sem_empty";

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int CHANNELS = 3; 
const int DATA_SIZE = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS;

// 이미지가 있는 폴더 경로 (수정 필요!)
const std::string IMG_DIR = "../../Documents/test_img"; 

int main() {
    // 1. 이미지 파일 목록 불러오기
    std::vector<std::string> image_files;
    
    if (!fs::exists(IMG_DIR)) {
        std::cerr << "Error: 폴더가 없습니다 -> " << IMG_DIR << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(IMG_DIR)) {
        std::string path = entry.path().string();
        // 확장자 필터링 (.jpg, .png 등)
        if (path.find(".jpg") != std::string::npos || path.find(".png") != std::string::npos) {
            image_files.push_back(path);
        }
    }

    // 순서대로 읽기 위해 정렬
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cerr << "Error: 폴더에 이미지 파일이 없습니다." << std::endl;
        return -1;
    }

    std::cout << ">>> " << image_files.size() << "장의 이미지를 로드했습니다." << std::endl;

    // 2. Shared Memory 생성
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, DATA_SIZE);
    unsigned char* shm_ptr = (unsigned char*)mmap(0, DATA_SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // 3. Semaphore 생성
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_EMPTY_NAME);
    sem_t* sem_full = sem_open(SEM_FULL_NAME, O_CREAT, 0666, 0);
    sem_t* sem_empty = sem_open(SEM_EMPTY_NAME, O_CREAT, 0666, 1);

    std::cout << ">>> Image Producer Started. Looping through images..." << std::endl;

    int idx = 0;
    cv::Mat frame, resized_frame, rgb_frame;
    int filesLength = image_files.size();

    while (true) {
        // [파일 읽기]
        std::string current_file = image_files[idx];
        ++idx;
        if (filesLength <= idx) {
            break;
        }
        frame = cv::imread(current_file);

        if (frame.empty()) {
            std::cerr << "Warning: 이미지를 읽을 수 없음: " << current_file << std::endl;
        } else {
            // [중요] Shared Memory 크기에 맞춰 리사이징
            cv::resize(frame, resized_frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
            
            // BGR -> RGB 변환
            cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

            // [Wait] Python이 읽어갈 때까지 대기
            sem_wait(sem_empty);

            // [Critical Section] 메모리 복사
            memcpy(shm_ptr, rgb_frame.data, DATA_SIZE);

            // [Signal] 데이터 준비 완료
            sem_post(sem_full);
            
            std::cout << "Sent: " << current_file << std::endl;
        }
        // [FPS 시뮬레이션]
        // 33ms 대기 = 약 30FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    // 정리 (실제로는 무한루프라 도달 안 함)
    munmap(shm_ptr, DATA_SIZE);
    close(shm_fd);
    shm_unlink(SHM_NAME);
    sem_close(sem_full);
    sem_close(sem_empty);
    
    return 0;
}