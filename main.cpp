#include <opencv2/opencv.hpp>
#include "MotionDetector.hpp"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    MotionDetector detector;
    cv::Mat frame, motionLog;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 모듈 사용
        motionLog = detector.process(frame);

        cv::imshow("Original", frame);
        cv::imshow("Motion Result", motionLog);

        if (cv::waitKey(30) == 'q') break;
    }

    return 0;
}