#include "SharedMemoryManager.hpp"  
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

int main() {
    const std::string img_dir = "../../Documents/test_img";
    
    SharedMemoryManager smm("yolo_frame", 640, 480);

    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        std::string path = entry.path().string();
        if (path.find(".jpg") != std::string::npos || path.find(".png") != std::string::npos) {
            image_files.push_back(path);
        }
    }
    std::sort(image_files.begin(), image_files.end());

    std::cout << ">>> Image Loop Started..." << std::endl;
    int filesLength = image_files.size();

    int idx = 0;
    while (true) {
        cv::Mat frame = cv::imread(image_files[idx]);
        ++idx;
        if (filesLength <= idx) {
            smm.sendExitSignal();
            break;
        }
        if (!frame.empty()) {
            smm.sendFrame(frame);
            std::cout << "Sent: " << image_files[idx] << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    return 0;
}