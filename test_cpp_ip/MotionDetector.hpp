#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <opencv2/opencv.hpp>

class MotionDetector {
public:
    MotionDetector();
    
    // 영상을 입력받아 차영상을 반환하는 핵심 함수
    cv::Mat process(const cv::Mat& inputFrame);

private:
    cv::Mat prevGray; // t-1 시점의 그레이스케일 이미지 저장
    bool isFirstFrame;
};

#endif