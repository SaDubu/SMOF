import pygame
import random
import os
import math

# 폴더 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "muti_view_simulation_logs")
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 상수 설정
CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)
NUM_SCENARIOS = 200
FPS = 60 # C++의 0.05초 주기에 맞추려면 실행 시 clock.tick(20) 조절 권장

class Object:
    def __init__(self, obj_id, is_main=False):
        self.original_id = obj_id
        self.id = obj_id
        self.is_main = is_main
        
        # 메인 객체는 사이드에서 시작, 노이즈는 아무데나
        if is_main:
            self.x = random.choice([random.uniform(20, 150), random.uniform(330, 450)])
        else:
            self.x = random.uniform(0, CANVAS_SIZE - 20)
            
        self.y = random.uniform(20, CANVAS_SIZE - 40)
        self.w = random.uniform(15, 25)
        self.h = random.uniform(15, 25)
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.has_entered = False

    def update(self):
        # 움직임 로직 (기존과 동일)
        self.vx += random.uniform(-0.2, 0.2)
        self.vy += random.uniform(-0.2, 0.2)
        
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 5:
            self.vx = (self.vx / speed) * 5
            self.vy = (self.vy / speed) * 5

        self.x += self.vx
        self.y += self.vy

        # 벽 튕기기
        if self.x < 0 or self.x + self.w > CANVAS_SIZE: self.vx *= -1
        if self.y < 0 or self.y + self.h > CANVAS_SIZE: self.vy *= -1

        # 타겟 영역 진입 체크 (메인 객체용)
        if TARGET_X_RANGE[0] < (self.x + self.w/2) < TARGET_X_RANGE[1]:
            self.has_entered = True

    def get_log_str(self):
        conf = random.uniform(0.7, 0.9)
        return f"{self.x:.4f}, {self.y:.4f}, {self.x+self.w:.4f}, {self.y+self.h:.4f}, {conf:.4f}, {self.id}"

def get_motion_region(main_obj):
    """메인 객체와 51% 이상 겹치는 랜덤 영역 생성"""
    # 51% 겹침을 보장하기 위해 크기를 비슷하게 하고 오프셋을 작게 줍니다.
    offset_limit = main_obj.w * 0.2  # 20% 이내로 움직이면 50% 이상 겹침이 보장됨
    m_w = main_obj.w * random.uniform(0.9, 1.1)
    m_h = main_obj.h * random.uniform(0.9, 1.1)
    m_x = main_obj.x + random.uniform(-offset_limit, offset_limit)
    m_y = main_obj.y + random.uniform(-offset_limit, offset_limit)
    return f"{m_x:.4f}, {m_y:.4f}, {m_x+m_w:.4f}, {m_y+m_h:.4f}"

# --- 실행 부분 ---
pygame.init()
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
font = pygame.font.SysFont("arial", 12)

for i in range(1, NUM_SCENARIOS + 1):
    main_obj = Object(i, is_main=True)
    noise_objs = [Object(random.randint(1000, 9999)) for _ in range(random.randint(1, 4))]
    
    scenario_logs = []
    frame_count = 0
    running = True

    while not main_obj.has_entered and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        screen.fill((0, 0, 0))
        # 배경 영역 표시
        pygame.draw.rect(screen, (0, 0, 150), (TARGET_X_RANGE[0], 0, 120, CANVAS_SIZE))

        # 1. 메인 객체 업데이트 및 로그
        main_obj.update()
        motion_str = get_motion_region(main_obj)
        
        # 프레임 기록 시작 (frame_idx, motion_x1, y1, x2, y2, num_detections)
        frame_header = f"FRAME: {frame_count}, MOTION: {motion_str}, OBJS:"
        obj_logs = [main_obj.get_log_str()]

        # 2. 노이즈 객체 업데이트 및 로그
        for n_obj in noise_objs:
            n_obj.update()
            obj_logs.append(n_obj.get_log_str())

        # 로그 저장
        scenario_logs.append(f"{frame_header} | {' / '.join(obj_logs)}")
        
        # 그리기
        main_obj.draw = lambda s, f: pygame.draw.rect(s, (255, 255, 255), (main_obj.x, main_obj.y, main_obj.w, main_obj.h), 2)
        main_obj.draw(screen, font)
        for n_obj in noise_objs:
            pygame.draw.rect(screen, (150, 150, 150), (n_obj.x, n_obj.y, n_obj.w, n_obj.h), 1)

        pygame.display.flip()
        pygame.time.Clock().tick(20) # 0.05초 주기에 맞춤
        frame_count += 1

    # 파일 저장
    if scenario_logs:
        with open(os.path.join(output_dir, f"scenario_{i:03d}.txt"), "w") as f:
            f.write("\n".join(scenario_logs))
    if not running: break

pygame.quit()