#include "MotionDetector.hpp"

cv::Mat MotionDetector::process(const cv::Mat& inputFrame) {
    cv::Mat currentGray, diff;

    cv::cvtColor(inputFrame, currentGray, cv::COLOR_BGR2GRAY);

    if (isFirstFrame) {
        currentGray.copyTo(prevGray);
        isFirstFrame = false;
        return cv::Mat::zeros(inputFrame.size(), CV_8UC1);
    }

    cv::absdiff(currentGray, prevGray, diff);

    currentGray.copyTo(prevGray);

    return diff;
}