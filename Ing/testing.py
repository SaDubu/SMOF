#이 파일은 아이디어를 테스트하는 파일임.
import cv2
import numpy as np
import smof_build as sb

vp = sb.VisionProcessor()
ma = sb.MotionAnalyzer()
od = sb.ObjectDetector()
nor_run = sb.run_experiment

#kal = sb.KalmanBoxTracker()
k_vp = sb.KalVisionProcessor()
k_ma = sb.KalMotionAnalyzer()
k_od = sb.KalObjectDetector()
k_run = sb.run_sort_experiment

video_cap = 0
video_option = True
IMG_SIZE = 640

if video_option :
    video_cap = None
    video_cap = sb.video_cap

def get_color(idx):
    np.random.seed(idx)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color

def grid_init(frame) :
    h, w = frame.shape
    p0 = ma.create_grid_points(h, w)
    return p0

def get_canvas(frame) :
    canvas = np.zeros_like(frame)
    return canvas

def get_midpoint(p1, p2) :
    return (p1 + p2) // 2

def get_rectangle_center(p1, p2) : 
    '''
    두 대각 점의 좌표를 이용하여 사각형의 중심점(Center Point)을 계산합니다.

    두 점 p1(x1, y1)과 p2(x2, y2)를 잇는 선분의 중점을 구함으로써 
    직사각형 혹은 선분의 중심 위치를 반환합니다.

    :param p1: 첫 번째 좌표 지점 (예: (x1, y1) 튜플)
    :param p2: 두 번째 좌표 지점 (예: (x2, y2) 튜플)
    :return: 중심점의 좌표 (x, y) 튜플
    '''
    x = get_midpoint(p1[0], p2[0])
    y = get_midpoint(p1[1], p2[1])
    center = (x, y)
    return center

def pred_target_pos(prev_pos, velocity, lost_count, frame_interval=0.03):
    """
    이전 위치와 속도를 바탕으로 객체의 현재 위치를 예측합니다. (등속도 모델)

    객체를 놓친 횟수(lost_count)가 많아질수록 가중치를 높여 
    마지막 이동 방향으로 더 멀리 예측 지점을 설정합니다.

    :param prev_pos: 마지막으로 확인된 위치 (x, y) 튜플 또는 리스트
    :param velocity: 객체의 이동 속도 벡터 (vx, vy)
    :param lost_count: 연속으로 객체를 놓친 프레임 수
    :param frame_interval: 프레임 간의 시간 간격 (기본값: 0.3)
    :return: 예측된 현재 위치 (pred_x, pred_y) 튜플
    """
    time_factor = (lost_count + 1) * frame_interval
    
    pred_x = prev_pos[0] + velocity[0] * time_factor
    pred_y = prev_pos[1] + velocity[1] * time_factor

    return (pred_x, pred_y)

def calculate_distance(p1, p2):
    """
    두 지점 p1과 p2의 거리 계산 피타고라스 이용

    :param p1: 첫 번째 지점의 좌표 (x1, y1)
    :param p2: 두 번째 지점의 좌표 (x2, y2)
    :return: 두 지점 사이의 직선 거리 (float)
    """
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist

def find_near_obj(detect_center_pos, detect_cls, objects, matched, min_dist=20):
    if len(objects) == 0 :
        return -1

    best_match_idx = -1
    
    for i, prev in enumerate(objects):
        if i in matched: continue
        if detect_cls != prev['cls']: continue

        pv = prev['vec'] if prev['vec'] is not None else (0, 0)
        pred_pos = pred_target_pos(prev['pos'], pv, prev['lost_count'])
        
        dist = calculate_distance(detect_center_pos, pred_pos)

        if dist < min_dist:
            best_match_idx = i
            min_dist = dist
        
    return best_match_idx

def draw_obj(frame, objects) :
    for obj in objects:
        if obj['lost_count'] == 0:
            ocx, ocy = obj['pos']
            
            # 벡터 화살표 그리기
            if obj.get('vec') is not None:
                avg_v = obj['vec']
                v_end = (int(ocx + avg_v[0] * 7), int(ocy + avg_v[1] * 7))
                cv2.arrowedLine(frame, (ocx, ocy), v_end, (0, 0, 255), 2)
            
            # [수정] 박스 그리기 (KeyError 방지를 위해 get() 사용 또는 조건문)
            if 'box' in obj:
                ox1, oy1, ox2, oy2 = obj['box']
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                
                # (선택) ID나 클래스 정보 표시
                cv2.putText(frame, f"obj_id:{obj['id']}", (ox1, oy1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return frame

def run():
    cap = cv2.VideoCapture(video_cap)

    start_one_time = True

    p0 = None
    trajectory_canvas = None
    prev_frame = None
    display_frame = None
    vectors = None
    detections = None
    matched_prev_indices = set()

    all_boxes = []
    curr_obj = []
    prev_obj = []
    next_id = 0
    threshold_value = 40
    max_lost = 15

    while True :
        ret, frame = cap.read()
        if not ret : return 1
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   

        if start_one_time :
            start_one_time = False
            p0 = grid_init(gray_frame)
            trajectory_canvas = get_canvas(frame)
            prev_frame = gray_frame
            continue

        display_frame = frame.copy()
        curr_obj = []

        diff, mask = vp.get_clean_mask(prev_frame, gray_frame, threshold_value)
        vectors = ma.get_vectors(prev_frame, gray_frame, p0, mask, 1.5, 25.0)
        detections = od.detect_moving_objects(frame, mask, 500, 50)

        all_boxes = [d['box'] for d in detections]

        for det in detections :
            x1, y1, x2, y2 = det['box']
            cx, cy = get_rectangle_center((x1, y1), (x2, y2))
            avg_v = ma.analyze_object_motion((x1, y1, x2, y2), all_boxes, vectors)

            best_match_id = find_near_obj((cx, cy), det['cls'], prev_obj, matched_prev_indices)

            if best_match_id != -1:
                obj = prev_obj[best_match_id]
                obj_id = obj['id']
                obj_color = obj['color']

                cv2.line(trajectory_canvas, obj['pos'], (cx, cy), obj_color, 2)
                matched_prev_indices.add(best_match_id)
            else:
                obj_id = next_id
                obj_color = get_color(obj_id)
                next_id += 1

            curr_obj.append({
                'id': obj_id,
                'pos': (cx, cy),
                'cls': det['cls'],
                'vec': avg_v,
                'lost_count': 0,
                'color': obj_color,
                'box': det['box']
            })
        
        for i, prev in enumerate(prev_obj):
            if i not in matched_prev_indices:
                prev['lost_count'] += 1
                curr_obj.append(prev)
            else :
                prev['lost_count'] = 0

        display_frame = draw_obj(display_frame, curr_obj)
        
        prev_obj =  curr_obj

        display_frame = vp.draw_contours(display_frame, mask)

        dot_mask = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY) > 0
        display_frame[dot_mask] = cv2.addWeighted(display_frame, 0.4, trajectory_canvas, 0.6, 0)[dot_mask]

        diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([diff_bgr, mask_bgr, display_frame])
        
        cv2.putText(combined, f"Thresh: {threshold_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        total_ids = len(prev_obj)
        cv2.putText(combined, f"Total IDs: {total_ids}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Integrated Modular Vision Machine', combined)

        # 루프 종료 및 제어
        if prev_frame is not None :
            prev_frame = cv2.addWeighted(prev_frame, 0.7, gray_frame, 0.3, 0)
        prev_frame = gray_frame.copy()
        key = cv2.waitKeyEx(30)
        if key == 27: break
        elif key == ord('c') or key == ord('C'):
            trajectory_canvas = np.zeros_like(trajectory_canvas)
        elif key in [0x260000, 24]: # Up
            threshold_value = min(255, threshold_value + 2)
        elif key in [0x280000, 25]: # Down
            threshold_value = max(0, threshold_value - 2)    

    return 0

if __name__ == '__main__' :
    run()