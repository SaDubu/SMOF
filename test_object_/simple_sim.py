import numpy as np
import os

# --- 1. 경로 설정 (현재 파일 기준) ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_dir = os.path.join(current_dir, "simulation_logs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"폴더 생성 완료: {output_dir}")

# --- 2. 설정값 ---
CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)  # 수정된 타겟 영역
NUM_SCENARIOS = 200

for i in range(1, NUM_SCENARIOS + 1):
    # 1. 초기화
    if np.random.rand() < 0.5:
        curr_x = np.random.uniform(0, TARGET_X_RANGE[0] - 30)
    else:
        curr_x = np.random.uniform(TARGET_X_RANGE[1] + 10, CANVAS_SIZE - 30)
        
    curr_y = np.random.uniform(0, CANVAS_SIZE - 30)
    curr_w = np.random.uniform(10, 30) 
    curr_h = np.random.uniform(10, 30)
    
    vx = np.random.uniform(-3, 3)
    vy = np.random.uniform(-3, 3)
    
    original_id = i
    curr_id = i
    log_data = []
    
    while True:
        # 매 프레임 속도 변경 및 이동 로직 (기존과 동일)
        accel_range = 0.2
        vx += np.random.uniform(-accel_range, accel_range)
        vy += np.random.uniform(-accel_range, accel_range)
        
        # 속도 제한
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 5.0:
            vx = (vx / speed) * 5.0; vy = (vy / speed) * 5.0
        elif speed < 1.0:
            vx = (vx / (speed + 1e-6)) * 1.5; vy = (vy / (speed + 1e-6)) * 1.5

        curr_x += vx
        curr_y += vy
        
        # 경계 처리
        if curr_x < 0 or curr_x + curr_w > CANVAS_SIZE:
            vx *= -1
            curr_x = np.clip(curr_x, 0, CANVAS_SIZE - curr_w)
        if curr_y < 0 or curr_y + curr_h > CANVAS_SIZE:
            vy *= -1
            curr_y = np.clip(curr_y, 0, CANVAS_SIZE - curr_h)
            
        # 노이즈 추가
        noise = lambda: np.random.uniform(0.00009, 0.009)
        curr_x += noise(); curr_y += noise(); curr_w += noise(); curr_h += noise()
        
        # ID 변경 로직 (기존과 동일)
        if curr_id != original_id:
            if np.random.rand() < 0.5: curr_id = original_id
        else:
            if np.random.rand() < 0.01: curr_id = np.random.randint(1, 201)

        # --- [수정 포인트] ---
        # 1. x1, y1, x2, y2 계산
        x1, y1 = curr_x, curr_y
        x2, y2 = curr_x + curr_w, curr_y + curr_h
        
        # 2. confidence 생성 (0.7 ~ 0.9)
        conf = np.random.uniform(0.7, 0.9)
        
        # 3. C++ Detection 구조체 순서: x1, y1, x2, y2, confidence, class_id
        # 구분자는 쉼표와 공백 한 칸으로 설정 (C++ ss >> 에서 처리하기 편함)
        log_str = f"{x1:.6f}, {y1:.6f}, {x2:.6f}, {y2:.6f}, {conf:.6f}, {curr_id}"
        log_data.append(log_str)

        # 종료 조건
        if TARGET_X_RANGE[0] < curr_x < TARGET_X_RANGE[1]:
            break

    # 파일 저장
    file_path = os.path.join(output_dir, f"scenario_{i:03d}.txt")
    with open(file_path, "w") as f:
        f.write("\n".join(log_data))

print(f"시뮬레이션 완료: 총 {NUM_SCENARIOS}개의 동적 이동 로그가 저장되었습니다.")