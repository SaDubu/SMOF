#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include "LockFreeQueue.h" // 작성하신 헤더 파일

int main() {
    const int num_threads = 8; // 오렌지 파이 5의 8코어 활용
    const int items_per_thread = 160;
    const int total_items = num_threads * items_per_thread;

    LockFreeQueue<int> queue;
    std::atomic<int> push_count(0);
    std::atomic<int> pop_count(0);
    std::atomic<long long> total_sum(0); // 데이터 무결성 체크용

    std::cout << "Starting Multi-thread Lock-Free Test..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 1. 생산자 스레드 시작
    std::vector<std::thread> producers;
    for (int i = 0; i < num_threads; ++i) {
        producers.emplace_back([&]() {
            for (int j = 0; j < items_per_thread; ++j) {
                queue.Push(j);
                push_count++;
            }
        });
    }

    // 2. 소비자 스레드 시작
    std::vector<std::thread> consumers;
    for (int i = 0; i < num_threads; ++i) {
        consumers.emplace_back([&]() {
            int local_count = 0;
            while (true) {
                auto val = queue.Pop();
                if (val) {
                    total_sum.fetch_add(*val);
                    pop_count++;
                    if (++local_count >= items_per_thread && pop_count >= total_items) break;
                } else {
                    if (pop_count >= total_items) break;
                    std::this_thread::yield(); // 큐가 비었으면 양보
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 3. 결과 출력
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << diff.count() << "s" << std::endl;
    std::cout << "Expected Pop: " << total_items << std::endl;
    std::cout << "Actual Pop  : " << pop_count.load() << std::endl;
    std::cout << "Final Size  : " << queue.Size() << " (Expected 0)" << std::endl;

    if (pop_count == total_items) {
        std::cout << ">>> SUCCESS: All items processed." << std::endl;
    } else {
        std::cout << ">>> FAILURE: Data loss detected!" << std::endl;
    }

    return 0;
}