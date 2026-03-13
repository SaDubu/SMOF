import pygame
import random
import os
import math

# 로그 저장 디렉토리
# 1. 현재 실행 중인 .py 파일의 절대 경로를 가져옵니다.
current_file_path = os.path.abspath(__file__)

# 2. 파일 이름 제외, 폴더 경로만 추출합니다.
current_dir = os.path.dirname(current_file_path)

# 3. 생성하고 싶은 폴더명을 결합합니다.
output_dir = os.path.join(current_dir, "view_simulation_logs")

# 4. 폴더가 없으면 생성합니다.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"폴더 생성 완료: {output_dir}")
else:
    print(f"이미 폴더가 존재합니다: {output_dir}")

# --- 2. 설정 (Constants) ---
CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)  # 변경된 타겟 영역
NUM_SCENARIOS = 200
FPS = 60

# 색상
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 150)
GRAY = (100, 100, 100) # 궤적 색상

pygame.init()
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
pygame.display.set_caption("Target Entry Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 12)

pygame.init()
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
pygame.display.set_caption("Movement Trace Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 12)

class Object:
    def __init__(self, obj_id):
        self.original_id = obj_id
        self.id = obj_id
        
        # 스폰 영역 제한 (중앙 제외)
        if random.random() < 0.5:
            self.x = random.uniform(20, TARGET_X_RANGE[0] - 30)
        else:
            self.x = random.uniform(TARGET_X_RANGE[1] + 10, CANVAS_SIZE - 40)
        
        self.y = random.uniform(20, CANVAS_SIZE - 40)
        self.w = random.uniform(15, 25)
        self.h = random.uniform(15, 25)
        
        # 초기 속도
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        
        self.log_data = []
        self.path = []
        self.has_entered = False

    def update(self):
        self.path.append((self.x + self.w/2, self.y + self.h/2))
        
        # --- [추가] 매 이동 시 속도 벡터 변경 (Random Acceleration) ---
        accel_range = 0.2
        self.vx += random.uniform(-accel_range, accel_range)
        self.vy += random.uniform(-accel_range, accel_range)
        
        # 속도가 너무 빨라지거나 멈추는 것을 방지 (Max Speed 5)
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 5:
            self.vx = (self.vx / speed) * 5
            self.vy = (self.vy / speed) * 5
        elif speed < 1: # 너무 느려지면 최소 속도 유지
            self.vx = (self.vx / (speed + 0.1)) * 1.5
            self.vy = (self.vy / (speed + 0.1)) * 1.5

        # 위치 이동
        self.x += self.vx
        self.y += self.vy

        # 벽 튕기기
        if self.x < 0 or self.x + self.w > CANVAS_SIZE:
            self.vx *= -1
            self.x = max(0, min(self.x, CANVAS_SIZE - self.w))
        if self.y < 0 or self.y + self.h > CANVAS_SIZE:
            self.vy *= -1
            self.y = max(0, min(self.y, CANVAS_SIZE - self.h))

        # 미세 노이즈 (기존 유지)
        noise = lambda: random.uniform(0.00009, 0.009)
        self.x += noise(); self.y += noise(); self.w += noise(); self.h += noise()

        # ID 변경 및 복구 로직
        if self.id != self.original_id:
            if random.random() < 0.5: self.id = self.original_id
        else:
            if random.random() < 0.01: self.id = random.randint(1, NUM_SCENARIOS)

        # 타겟 영역 진입 체크
        if TARGET_X_RANGE[0] < self.x < TARGET_X_RANGE[1]:
            self.has_entered = True

    def draw(self):
        if len(self.path) > 2:
            pygame.draw.lines(screen, GRAY, False, self.path, 1)
        rect = pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))
        pygame.draw.rect(screen, WHITE, rect, 1)
        id_txt = font.render(str(self.id), True, WHITE)
        screen.blit(id_txt, (int(self.x), int(self.y) - 15))

    def log(self):
        self.log_data.append(f"{self.id}, {self.x:.6f}, {self.y:.6f}, {self.w:.6f}, {self.h:.6f}")

# --- 메인 루프 ---
running = True
for i in range(1, NUM_SCENARIOS + 1):
    obj = Object(i)
    filename = f"scenario_{i:03d}.txt"
    
    while not obj.has_entered and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        screen.fill(BLACK)
        target_rect = pygame.Rect(TARGET_X_RANGE[0], 0, TARGET_X_RANGE[1] - TARGET_X_RANGE[0], CANVAS_SIZE)
        pygame.draw.rect(screen, BLUE, target_rect)

        obj.update()
        obj.log()
        obj.draw()

        pygame.display.flip()
        clock.tick(FPS)

    if obj.log_data:
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("\n".join(obj.log_data))
    if not running: break

pygame.quit()