#ifndef SHARED_MEMORY_MANAGER_HPP
#define SHARED_MEMORY_MANAGER_HPP

#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>
#include <opencv2/opencv.hpp>

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

public:
    SharedMemoryManager(std::string name, int w, int h, int c = 3) 
        : shm_name("/" + name + "_shm"), 
          sem_full_name("/" + name + "_full"), 
          sem_empty_name("/" + name + "_empty"),
          sem_exit_name("/" + name + "_exit"),
          width(w), height(h), channels(c) {
        
        data_size = width * height * channels;

        // Shared Memory 생성
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        ftruncate(shm_fd, data_size);
        shm_ptr = (unsigned char*)mmap(0, data_size, PROT_WRITE, MAP_SHARED, shm_fd, 0);

        // Semaphore 초기화
        sem_unlink(sem_full_name.c_str());
        sem_unlink(sem_empty_name.c_str());
        sem_full = sem_open(sem_full_name.c_str(), O_CREAT, 0666, 0);
        sem_empty = sem_open(sem_empty_name.c_str(), O_CREAT, 0666, 1);
        sem_exit = sem_open(sem_exit_name.c_str(), O_CREAT, 0666, 0);
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
};

#endif