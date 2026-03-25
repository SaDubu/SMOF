#ifndef LABOR_MANAGER_HPP
#define LABOR_MANAGER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>

#include "define.h"

class MotionDetector; 
template <typename T> class LockFreeQueueSPSC;
class SharedMemoryManager;

class LaborManager {
public :
    LaborManager(bool* is_running_ptr);

    void capture_worker(std::string& pipe, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& display_q);

    void dmr_worker(MotionDetector& detector, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& motion_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q);

    void diff_worker(MotionDetector& detector, LockFreeQueueSPSC<cv::Mat>& raw_q, LockFreeQueueSPSC<cv::Mat>& motion_q);

    void mask_worker(LockFreeQueueSPSC<cv::Mat>& motion_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& mask_q);

    void rect_worker(LockFreeQueueSPSC<cv::Mat>& mask_q, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q);

    void draw_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& bbox_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& final_q);

    void RGB_draw_save_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& bbox_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& final_q, LockFreeQueueSPSC<LockFreeQueueSPSC<std::string>*>& file_lists);

    void crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<std::vector<cv::Mat>>& chips_q, LockFreeQueueSPSC<ChipInfo>& o_q);

    void new_crop_worker(LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<cv::Mat>& display_q, LockFreeQueueSPSC<cv::Mat>& filtered_frame_q);

    void track_worker(LockFreeQueueSPSC<std::vector<Detection>>& objects_q, TrackerVector* trackers);

    void yolo_worker(SharedMemoryManager& smm, LockFreeQueueSPSC<cv::Mat>& frames, LockFreeQueueSPSC<std::vector<Detection>>& yolo_results);

    void filter_worker(LockFreeQueueSPSC<std::vector<Detection>>& yolo_results, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q, LockFreeQueueSPSC<std::vector<Detection>>& filter_results);

    void get_image_move_area(LockFreeQueueSPSC<LockFreeQueueSPSC<std::string>*>& file_lists, LockFreeQueueSPSC<cv::Mat>& frames, LockFreeQueueSPSC<std::vector<cv::Rect>>& rect_q);

private :
    bool* m_is_running;

    bool mask_moving_area(cv::Mat& motion_image, cv::Mat& result);

    std::vector<cv::Rect> merge_boxes(std::vector<cv::Rect>& rects);

    std::vector<cv::Rect> get_boxes(cv::Mat& mask);
};

#endif