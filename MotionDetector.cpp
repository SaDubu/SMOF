#include "MotionDetector.hpp"

MotionDetector::MotionDetector() : isFirstFrame(true) {}

cv::Mat MotionDetector::process(const cv::Mat& inputFrame) {
    cv::Mat currentGray, diff;
    
    // 1. 그레이스케일 변환
    cv::cvtColor(inputFrame, currentGray, cv::COLOR_BGR2GRAY);

    // 2. 첫 프레임 처리 (비교 대상이 없으므로 빈 영상 반환)
    if (isFirstFrame) {
        currentGray.copyTo(prevGray);
        isFirstFrame = false;
        return cv::Mat::zeros(inputFrame.size(), CV_8UC1);
    }

    // 3. 차영상 계산 (t - (t-1))
    cv::absdiff(currentGray, prevGray, diff);

    // 4. 다음 연산을 위해 현재 프레임 저장
    currentGray.copyTo(prevGray);

    return diff;
}