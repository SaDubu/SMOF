#include <iostream>
#include <cstdlib>
#include <ctime>

//대충 struct 만들어서 테스트 해봄.

struct History {
    int history[30] = {0};
    int head = 0;
    int current_size = 0;

    int values[30] = {0};
    int freqs[30] = {0};
    int unique_count = 0;

    size_t opti_count = 0;
    size_t brute_count = 0;

    void add(int class_id) { 
        history[head] = class_id; 
        head = (head + 1) % 30; 
        if (current_size < 30) ++current_size; 
    }

    int get_mode() {
        int max_val = history[0];
        int max_cnt = 0;

        for (int i = 0; i < current_size; ++i) {
            int current = history[i];

            bool seen = false;
            for (int k = 0; k < i; ++k) {
                if (history[k] == current) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;

            int current_cnt = 0;
            for (int j = 0; j < current_size; ++j) {
                ++opti_count;
                if (history[j] == current) current_cnt++;
            }

            if (current_cnt > max_cnt) {
                max_cnt = current_cnt;
                max_val = current;
            }
        }

        return max_val;
    }

    // 검증용 Brute Force (O(n^2))
    int brute_mode() {
        if (current_size == 0) return -1;
        
        int max_val = -1;
        int max_cnt = -1;

        for (int i = 0; i < current_size; ++i) {
            int current = history[i];
            int current_cnt = 0;
            for (int j = 0; j < current_size; ++j) {
                ++brute_count;
                if (history[j] == current) current_cnt++;
            }

            if (current_cnt > max_cnt) {
                max_cnt = current_cnt;
                max_val = current;
            }
        }
        return max_val;
    }
};

int main() {

    srand(1);

    History h;

    for (int i = 0; i < 100000000; i++) {

        int val = rand() % 103;

        h.add(val);

        int a = h.get_mode();
        int b = h.brute_mode();

        if (a != b) {
            std::cout << " Mismatch " << a << " opti " << b << " O(n^2)" << std::endl;
        }

        if (i % 100000 == 0) {
            std::cout << "i=" << i << " ";
            std::cout << "opti_count=" << h.opti_count << ", brute_count=" << h.brute_count;
            std::cout << " mode=" << h.get_mode() << "\n";
        }
    }

    std::cout << "All tests passed\n";
}