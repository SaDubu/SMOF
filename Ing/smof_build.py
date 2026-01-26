import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

#지금 구축하고 있는 방식의 완성본이 될 코드이므로 모듈화에 조금 더 집중해서 진행을 해야한다.
#일단 모듈화로 이루는 것은 나중에 진행해도 별 문제 없을 것 같고 이건 일단 기본적으로 아이디어 검증을 해야한다고 생각함.
#이걸 잘 만들어야 나중에 모듈화를 시킬 때 큰 무리 없이 진행이 가능할 것 같음.

class VisionProcessor:
    def __init__(self, kernel_size=(11, 11)):
        # 타원형 커널을 사용하여 더 부드러운 덩어리 효과를 줍니다.
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)

    def get_morph_mask(self, prev_gray, curr_gray, thresh_val):
        """차분 및 모폴로지를 통해 정제된 마스크 생성"""
        # 노이즈 제거를 위한 블러
        p_blur = cv.GaussianBlur(prev_gray, (5, 5), 0)
        c_blur = cv.GaussianBlur(curr_gray, (5, 5), 0)
        
        diff = cv.absdiff(c_blur, p_blur)
        _, thresh = cv.threshold(diff, thresh_val, 255, cv.THRESH_BINARY)
        
        # 모폴로지 연산 (CLOSE로 구멍 메우기)
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, self.kernel)    
        morph = cv.dilate(morph, self.kernel, iterations=1)
        return diff, morph

    def draw_contours(self, frame, mask, color=(0, 255, 0)):
        """마스크에서 윤곽선을 추출하여 프레임에 그림"""
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame, contours, -1, color, 1)
        return frame

    def get_clean_mask(self, prev_gray, curr_gray, thresh_val):
        """
        사용자 제안 방식: 중간값 필터와 이중 이진화를 이용한 강력한 윤곽선 마스크 생성
        """
        # 1. 노이즈 억제 (중간값 필터)
        p_blur = cv.medianBlur(prev_gray, 7)
        c_blur = cv.medianBlur(curr_gray, 7)
        
        # 2. 차분 및 1차 임계값 적용
        diff = cv.absdiff(c_blur, p_blur)
        _, thresh = cv.threshold(diff, thresh_val, 255, cv.THRESH_BINARY)
        
        # 3. 모폴로지 스택 (강력한 덩어리화)
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, self.kernel, iterations=2)
        
        # 4. [핵심] 가우시안 블러 후 재이진화 (경계선 매끄럽게 다듬기)
        morph = cv.GaussianBlur(morph, (15, 15), 0)
        _, refined_mask = cv.threshold(morph, 127, 255, cv.THRESH_BINARY)
        
        return diff, refined_mask

    def draw_refined_contours(self, frame, mask, min_area=1000):
        """
        다각형 근사화를 적용한 매끄러운 윤곽선 그리기
        """
        # 정교한 윤곽선 추출 알고리즘 사용
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        
        for cnt in contours:
            if cv.contourArea(cnt) > min_area:
                # 윤곽선 다듬기 (단순화)
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.005 * peri, True)
                # 선명한 녹색 윤곽선
                cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        return frame
    
class MotionAnalyzer:
    def __init__(self, grid_step=10):
        self.grid_step = grid_step
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def create_grid_points(self, h, w):
        """격자 기반의 초기 포인트 생성"""
        gy, gx = np.mgrid[self.grid_step//2:h:self.grid_step, 
                          self.grid_step//2:w:self.grid_step].reshape(2, -1)
        return np.stack((gx, gy), axis=1).astype(np.float32).reshape(-1, 1, 2)

    def get_vectors(self, prev_gray, curr_gray, p0, mask, min_d, max_d):
        """광학 흐름 계산 및 움직임 영역 내 유효 벡터 필터링"""
        p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        vectors = []
        if p1 is not None:
            for i, (new, old) in enumerate(zip(p1, p0)):
                if st[i]:
                    nx, ny, ox, oy = *new.ravel(), *old.ravel()
                    # 움직임 마스크(흰색 영역) 내부의 벡터만 인정
                    if 0 <= int(oy) < mask.shape[0] and 0 <= int(ox) < mask.shape[1] and mask[int(oy), int(ox)] == 255:
                        dx, dy = nx - ox, ny - oy
                        if min_d < np.sqrt(dx**2 + dy**2) < max_d:
                            vectors.append({'pos': (int(ox), int(oy)), 'vec': (dx, dy)})
        return vectors

    def analyze_object_motion(self, target_box, all_boxes, all_vectors):
        """
        target_box: (x1, y1, x2, y2) 형태의 좌표 튜플
        all_boxes: 현재 프레임의 모든 객체 박스 리스트
        all_vectors: 전체 광학 흐름 벡터
        """
        # [수정 포인트] target_det['box'] 대신 인자로 받은 target_box를 직접 사용
        tx1, ty1, tx2, ty2 = target_box
        area = (tx2 - tx1) * (ty2 - ty1)
        
        # 1. 타겟 박스 내부 벡터 추출
        # vectors 구조가 [{'pos': (x,y), 'vec': (dx,dy)}, ...] 형태임에 유의
        candidate_vecs_data = [v for v in all_vectors if tx1 <= v['pos'][0] <= tx2 and ty1 <= v['pos'][1] <= ty2]
        
        # 2. 내부 겹침 박스 제외 로직
        final_vecs = []
        for v_data in candidate_vecs_data:
            px, py = v_data['pos']
            is_inside_other = False
            for other_box in all_boxes:
                # 자기 자신(현재 target_box)과 좌표가 완전히 같으면 스킵
                if np.array_equal(other_box, target_box):
                    continue
                
                ox1, oy1, ox2, oy2 = other_box
                # 다른 박스 안에 벡터 포인트가 포함되는지 검사
                if ox1 <= px <= ox2 and oy1 <= py <= oy2:
                    is_inside_other = True
                    break
            
            if not is_inside_other:
                final_vecs.append(v_data['vec'])

        # [필터링] 유효 벡터 밀도 검사
        if len(final_vecs) < 5 or (len(final_vecs) / (area + 1e-5) < 0.001):
            return None

        # [필터링] 평균 이동 강도 검사
        avg_v = self._calculate_weighted_average(final_vecs)
        if avg_v is not None:
            magnitude = np.sqrt(avg_v[0]**2 + avg_v[1]**2)
            if magnitude < 2.0:  # 노이즈 제거
                return None
                
        return avg_v

    def _is_inside(self, inner, outer):
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer
        return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

    def _calculate_weighted_average(self, vecs):
        """벡터 집합의 가중 평균 계산 (내부 로직)"""
        if len(vecs) < 3: return None
        
        vec_array = np.array(vecs)
        center = np.median(vec_array, axis=0)
        dists = np.linalg.norm(vec_array - center, axis=1)
        
        sigma = np.std(dists) + 1e-5
        weights = np.exp(-0.5 * (dists / sigma)**2)
        
        if np.sum(weights) > 1e-5:
            return (np.average(vec_array[:, 0], weights=weights), 
                    np.average(vec_array[:, 1], weights=weights))
        mean_res = np.mean(vec_array, axis=0)
        return (float(mean_res[0]), float(mean_res[1]))
    

class ObjectDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect_moving_objects(self, frame, mask, n_limit, m_limit):
        if cv.countNonZero(mask) < n_limit: return []
        results = self.model(frame, verbose=False)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, cls in zip(boxes, clss):
                x1, y1, x2, y2 = map(int, box)
                roi = mask[max(0, y1):y2, max(0, x1):x2]
                if roi.size > 0:
                    overlap_ratio = cv.countNonZero(roi) / roi.size
                    if overlap_ratio >= 0.1 and cv.countNonZero(roi) >= m_limit:
                        detections.append({'box': (x1, y1, x2, y2), 'cls': cls})
        return detections

# --- [메인 컨트롤러] ---
def run_experiment():
    # 1. 환경 설정
    cap = cv.VideoCapture('../datasets/MOT15/train/PETS09-S2L1/img1/%06d.jpg')
    #cap = cv.VideoCapture(0)
    vp = VisionProcessor()
    ma = MotionAnalyzer(grid_step=10)
    od = ObjectDetector()
    
    ret, old_frame = cap.read()
    if not ret: return
    
    prev_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    p0 = ma.create_grid_points(h, w)
    
    trajectory_canvas = np.zeros_like(old_frame)
    threshold_value = 40

    print("--- 실험 시작 ---")
    print("방향키 ↑/↓: 임계값 조절 | 'c': 궤적 초기화 | ESC: 종료")
    max_lost_frames = 15  # [설정] 최대 n-프레임까지 소실 허용
    prev_objects = []     # [{'pos':(x,y), 'cls':c, 'vec':(dx,dy), 'lost_count':0}]

    next_id = 0

    min_dist = 20 # 매칭 반경

    def get_color(idx):
        # ID를 기반으로 고정된 랜덤 색상 생성 (BGR)
        np.random.seed(idx)
        return tuple(map(int, np.random.randint(0, 255, 3)))

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 2. 모듈별 로직 수행
        diff, mask = vp.get_clean_mask(prev_gray, curr_gray, threshold_value)
        vectors = ma.get_vectors(prev_gray, curr_gray, p0, mask, 1.5, 25.0)
        detections = od.detect_moving_objects(frame, mask, 500, 50)

        all_boxes = [d['box'] for d in detections]
        curr_objects = []
        matched_prev_indices = set()

        # 1. 현재 탐지된 객체들 탐색
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            avg_v = ma.analyze_object_motion((x1, y1, x2, y2), all_boxes, vectors)

            best_match_idx = -1

            # 2. 보관 중인 이전 객체들과 매칭 (가장 가까운 객체 찾기)
            for i, prev in enumerate(prev_objects):
                if i in matched_prev_indices: continue
                if det['cls'] != prev['cls']: continue

                pv = prev['vec'] if prev['vec'] is not None else (0, 0)
                pred_x = prev['pos'][0] + pv[0] * ((prev['lost_count'] + 1) * 0.3)
                pred_y = prev['pos'][1] + pv[1] * ((prev['lost_count'] + 1) * 0.3)

                dist = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                if dist < min_dist:
                    best_match_idx = i

            if best_match_idx != -1:
                # [매칭 성공] 기존 ID와 색상을 계승
                prev_obj = prev_objects[best_match_idx]
                obj_id = prev_obj['id']
                obj_color = prev_obj['color']
                
                # 고유 색상으로 궤적 그리기
                cv.line(trajectory_canvas, prev_obj['pos'], (cx, cy), obj_color, 2)
                matched_prev_indices.add(best_match_idx)
            else:
                # [매칭 실패] 신규 ID 부여 및 색상 생성
                obj_id = next_id
                obj_color = get_color(obj_id)
                next_id += 1

            # curr_objects에 모든 정보 저장 (box 포함!)
            curr_objects.append({
                'id': obj_id,
                'pos': (cx, cy),
                'cls': det['cls'],
                'vec': avg_v,
                'lost_count': 0,
                'color': obj_color,
                'box': (x1, y1, x2, y2)
            })
            

        # 3. 이번 프레임에서 탐지되지 않은(매칭 안 된) 이전 객체들 처리
        for i, prev in enumerate(prev_objects):
            if i not in matched_prev_indices:
                if prev['lost_count'] < max_lost_frames:
                    # 위치 관성 이동
                    pv = prev['vec'] if prev['vec'] is not None else (0, 0)
                    new_pos = (int(prev['pos'][0] + pv[0]), int(prev['pos'][1] + pv[1]))
                    
                    # [수정] 박스 좌표도 이동 방향에 따라 같이 옮겨주면 좋습니다.
                    if 'box' in prev:
                        bx1, by1, bx2, by2 = prev['box']
                        prev['box'] = (int(bx1 + pv[0]), int(by1 + pv[1]), 
                                       int(bx2 + pv[0]), int(by2 + pv[1]))
                    
                    prev['pos'] = new_pos
                    prev['lost_count'] += 1
                    curr_objects.append(prev) # 소실 카운트가 올라간 상태로 보존
            else :
                prev['lost_count'] = 0

        # 4. 시각화
        for obj in curr_objects:
            # 실제로 이번 프레임에 탐지된 객체만 화면에 박스와 화살표를 그림
            if obj['lost_count'] == 0:
                ocx, ocy = obj['pos']
                
                # 벡터 화살표 그리기
                if obj.get('vec') is not None:
                    avg_v = obj['vec']
                    v_end = (int(ocx + avg_v[0] * 7), int(ocy + avg_v[1] * 7))
                    cv.arrowedLine(display_frame, (ocx, ocy), v_end, (0, 0, 255), 2)
                
                # [수정] 박스 그리기 (KeyError 방지를 위해 get() 사용 또는 조건문)
                if 'box' in obj:
                    ox1, oy1, ox2, oy2 = obj['box']
                    cv.rectangle(display_frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                    
                    # (선택) ID나 클래스 정보 표시
                    cv.putText(display_frame, f"obj_id:{obj['id']}", (ox1, oy1-10), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_objects = curr_objects

        # 배경 업데이트 및 캔버스 합성
        vp.draw_contours(display_frame, mask)

        # 4. 화면 합성 및 출력
        dot_mask = cv.cvtColor(trajectory_canvas, cv.COLOR_BGR2GRAY) > 0
        display_frame[dot_mask] = cv.addWeighted(display_frame, 0.4, trajectory_canvas, 0.6, 0)[dot_mask]
        
        #diff_bgr = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
        #mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        #combined = cv.hconcat([diff_bgr, mask_bgr, display_frame])

        combined = display_frame
        
        cv.putText(combined, f"Thresh: {threshold_value}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        total_ids = len(prev_objects)
        cv.putText(combined, f"Total IDs: {total_ids}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow('Integrated Modular Vision Machine', combined)

        # 루프 종료 및 제어
        if prev_gray is not None :
            prev_gray = cv.addWeighted(prev_gray, 0.7, curr_gray, 0.3, 0)
        prev_gray = curr_gray.copy()
        key = cv.waitKeyEx(30)
        if key == 27: break
        elif key == ord('c') or key == ord('C'):
            trajectory_canvas = np.zeros_like(old_frame)
        elif key in [0x260000, 24]: # Up
            threshold_value = min(255, threshold_value + 2)
        elif key in [0x280000, 25]: # Down
            threshold_value = max(0, threshold_value - 2)

    cap.release()
    cv.destroyAllWindows()

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # 상태 변수: [x, y, s, r, vx, vy, vs] (중심x, 중심y, 넓이, 종횡비, 각 변화율)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.z_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self):
        return self.z_to_bbox(self.kf.x)[0]

    def bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def z_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))
        
def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    iou = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) 
          + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return iou


# --- [기존 모듈: Vision & Motion] ---
class KalVisionProcessor:
    def __init__(self, kernel_size=(11, 11)):
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)

    def get_clean_mask(self, prev_gray, curr_gray, thresh_val):
        p_blur = cv.medianBlur(prev_gray, 7)
        c_blur = cv.medianBlur(curr_gray, 7)
        diff = cv.absdiff(c_blur, p_blur)
        _, thresh = cv.threshold(diff, thresh_val, 255, cv.THRESH_BINARY)
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, self.kernel, iterations=2)
        morph = cv.GaussianBlur(morph, (15, 15), 0)
        _, refined_mask = cv.threshold(morph, 127, 255, cv.THRESH_BINARY)
        return refined_mask

class KalMotionAnalyzer:
    def __init__(self, grid_step=10):
        self.grid_step = grid_step
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    def create_grid_points(self, h, w):
        gy, gx = np.mgrid[self.grid_step//2:h:self.grid_step, 
                          self.grid_step//2:w:self.grid_step].reshape(2, -1)
        return np.stack((gx, gy), axis=1).astype(np.float32).reshape(-1, 1, 2)

    def get_vectors(self, prev_gray, curr_gray, p0, mask):
        p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        vectors = []
        if p1 is not None:
            for i, (new, old) in enumerate(zip(p1, p0)):
                if st[i]:
                    nx, ny, ox, oy = *new.ravel(), *old.ravel()
                    if 0 <= int(oy) < mask.shape[0] and 0 <= int(ox) < mask.shape[1] and mask[int(oy), int(ox)] == 255:
                        dx, dy = nx - ox, ny - oy
                        if 1.5 < np.sqrt(dx**2 + dy**2) < 25.0:
                            vectors.append({'pos': (int(ox), int(oy)), 'vec': (dx, dy)})
        return vectors

class KalObjectDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame, mask):
        results = self.model(frame, verbose=False)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            conf = results[0].boxes.conf.cpu().numpy()
            for box, cls, c in zip(boxes, clss, conf):
                if cls == 0 and c > 0.3: # Person class
                    detections.append(box)
        return np.array(detections)

def run_sort_experiment():
    video_path = '../datasets/MOT15/train/PETS09-S2L1/img1/%06d.jpg'
    output_path = 'output/PETS09-S2L1_SORT.txt'
    
    cap = cv.VideoCapture(video_path)
    vp = KalVisionProcessor()
    ma = KalMotionAnalyzer()
    od = KalObjectDetector()

    trackers = []
    max_age = 20  # 객체 소멸 대기 프레임
    min_hits = 3  # 유효 객체 판단 최소 프레임
    iou_threshold = 0.3

    frame_idx = 0
    mot_results = []
    
    ret, old_frame = cap.read()
    prev_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = ma.create_grid_points(*prev_gray.shape)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        curr_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # 1. 탐지 및 전처리
        mask = vp.get_clean_mask(prev_gray, curr_gray, 40)
        detections = od.detect(frame, mask) # [x1, y1, x2, y2] 배열
        
        # 2. 트래커 예측(Predict)
        trks = np.zeros((len(trackers), 5))
        to_del = []
        for t, trk in enumerate(trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trackers = [t for i, t in enumerate(trackers) if i not in to_del]
        trks = np.delete(trks, to_del, axis=0)

        # 3. 매칭(Hungarian Matching)
        matched, unmatched_dets, unmatched_trks = [], [], []
        if len(detections) > 0 and len(trks) > 0:
            iou_matrix = iou_batch(detections, trks)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            for d, t in zip(row_ind, col_ind):
                if iou_matrix[d, t] < iou_threshold:
                    unmatched_dets.append(d)
                    unmatched_trks.append(t)
                else:
                    matched.append((d, t))
            unmatched_dets += [d for d in range(len(detections)) if d not in row_ind]
            unmatched_trks += [t for t in range(len(trks)) if t not in col_ind]
        else:
            unmatched_dets = list(range(len(detections)))

        # 4. 트래커 업데이트(Update)
        for d, t in matched:
            trackers[t].update(detections[d])

        # 5. 신규 객체 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            trackers.append(trk)

        # 6. 결과 정리 및 시각화
        i = len(trackers)
        for trk in reversed(trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= min_hits or frame_idx <= min_hits):
                # MOT 저장 (ID는 1부터 시작하도록 조정)
                res_id = trk.id + 1
                x1, y1, x2, y2 = d
                mot_results.append({'frame': frame_idx, 'id': res_id, 'box': (x1, y1, x2, y2)})
                
                # 시각화
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv.putText(frame, f"ID:{res_id}", (int(x1), int(y1)-10), 0, 0.6, (255, 255, 0), 2)

            i -= 1
            if trk.time_since_update > max_age:
                trackers.pop(i)

        cv.imshow('SORT Enhanced Tracker', frame)
        if cv.waitKey(1) == 27: break
        prev_gray = curr_gray.copy()

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for r in mot_results:
            w, h = r['box'][2] - r['box'][0], r['box'][3] - r['box'][1]
            f.write(f"{r['frame']},{r['id']},{r['box'][0]:.2f},{r['box'][1]:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    
    cap.release()
    cv.destroyAllWindows()
    print(f"저장 완료: {output_path}")