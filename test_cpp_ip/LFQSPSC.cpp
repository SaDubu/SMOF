#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>

#include "LFQSPSC.h"

/**
 * @brief SPSC 큐의 무결성을 테스트하는 함수
 * @param test_count 테스트할 데이터 개수 (예: 1,000,000)
 */
void TestSPSCQueue(const int test_count) {
    LockFreeQueueSPSC<int> queue;
    
    long long push_sum = 0;
    long long pop_sum = 0;
    int pop_count = 0;

    std::cout << "--- SPSC Queue Test Start (" << test_count << " items) ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::thread producer([&]() {
        for (int i = 1; i <= test_count; ++i) {
            push_sum += i;
            queue.Push(i);
        }
    });

    std::thread consumer([&]() {
        int value;
        while (pop_count < test_count) {
            if (queue.Pop(value)) {
                pop_sum += value;
                pop_count++;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Elapsed Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Push Sum: " << push_sum << " | Pop Sum: " << pop_sum << std::endl;
    std::cout << "Processed Items: " << pop_count << std::endl;

    if (push_sum == pop_sum && pop_count == test_count) {
        std::cout << ">>> RESULT: SUCCESS (No data loss or corruption)" << std::endl;
    } else {
        std::cout << ">>> RESULT: FAILURE (Data mismatch detected!)" << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
}