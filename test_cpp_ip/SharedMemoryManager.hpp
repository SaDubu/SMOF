#ifndef SHARED_MEMORY_MANAGER_HPP
#define SHARED_MEMORY_MANAGER_HPP

#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <opencv2/opencv.hpp>

#include "define.h"

class SharedMemoryManager {
private:
    std::string shm_name;
    std::string sem_full_name;
    std::string sem_empty_name;
    std::string sem_exit_name;
    int width, height, channels;
    size_t data_size;

    int shm_fd;
    unsigned char* shm_ptr;
    sem_t *sem_full, *sem_empty, *sem_exit;

    std::string yolo_shm_name;
    std::string yolo_sem_name;
    int yolo_fd;
    void* yolo_ptr;
    sem_t *sem_yolo;
    size_t yolo_size;

    uint16_t last_seq = 0;
    bool is_done = false;

    std::vector<Detection> results;

    size_t debug_count = 0;

public:
    SharedMemoryManager(std::string name, int w, int h, int c = 3) 
        : shm_name("/" + name + "_shm"), 
          sem_full_name("/" + name + "_full"), 
          sem_empty_name("/" + name + "_empty"),
          sem_exit_name("/" + name + "_exit"),
          yolo_shm_name("/" + name + "_yolo"),  
          yolo_sem_name("/" + name + "_yolo_sem"),
          width(w), height(h), channels(c) {
        
        data_size = width * height * channels;
        yolo_size = 4 + (100 * sizeof(Detection));

        // Shared Memory 생성
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        ftruncate(shm_fd, data_size);
        shm_ptr = (unsigned char*)mmap(0, data_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);

        // Semaphore 초기화
        sem_unlink(sem_full_name.c_str());
        sem_unlink(sem_empty_name.c_str());
        sem_unlink(sem_exit_name.c_str());
        sem_full = sem_open(sem_full_name.c_str(), O_CREAT, 0666, 0);
        sem_empty = sem_open(sem_empty_name.c_str(), O_CREAT, 0666, 1);
        sem_exit = sem_open(sem_exit_name.c_str(), O_CREAT, 0666, 0);
        
        sem_unlink(yolo_sem_name.c_str());
        yolo_fd = shm_open(yolo_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        ftruncate(yolo_fd, yolo_size);
        yolo_ptr = mmap(0, yolo_size, PROT_READ | PROT_WRITE, MAP_SHARED, yolo_fd, 0);

        sem_yolo = sem_open(yolo_sem_name.c_str(), O_CREAT, 0666, 1);
    }

    // 객체가 사라질 때 자동으로 리소스 정리
    ~SharedMemoryManager() {
        munmap(shm_ptr, data_size);
        close(shm_fd);
        shm_unlink(shm_name.c_str());
        sem_close(sem_full);
        sem_close(sem_empty);
        sem_unlink(sem_full_name.c_str());
        sem_unlink(sem_empty_name.c_str());

        munmap(yolo_ptr, yolo_size);
        close(yolo_fd);
        shm_unlink(yolo_shm_name.c_str());
        sem_close(sem_yolo);
        sem_unlink(yolo_sem_name.c_str());
        std::cout << "SharedMemory Resources Cleaned Up." << std::endl;
    }

    void sendFrame(const cv::Mat& frame) {
        if (frame.empty()) return;

        cv::Mat resized, rgb;
        cv::resize(frame, resized, cv::Size(width, height));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        sem_wait(sem_empty);
        memcpy(shm_ptr, rgb.data, data_size);
        sem_post(sem_full);
    }
    
    void sendExitSignal() {
        sem_post(sem_exit);
    }

    std::vector<Detection>* receiveYoloResult() {
        if (yolo_ptr == MAP_FAILED) {
            perror("mmap failed");
            return nullptr;
        }
        
        sem_wait(sem_yolo);
        uint16_t current_seq = *(uint16_t*)yolo_ptr;

        if (current_seq == last_seq) {
            sem_post(sem_yolo);
            return nullptr; 
        }

        results.clear();

        int count = *(int*)((char*)yolo_ptr + 4);
        //printf("\n감지된 객체 수: %d\n", count);
        //++debug_count;
        //printf("\n%ld\n", debug_count);

        if (count > 0 && count <= 100) {
            Detection* data_ptr = (Detection*)((char*)yolo_ptr + 8);
            
            results.assign(data_ptr, data_ptr + count);

            is_done = true;
        }

        last_seq = current_seq;

        sem_post(sem_yolo);
        return &results;
    }
};

#endif