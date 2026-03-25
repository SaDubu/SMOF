import pygame
import random
import os
import math

# ... [앞부분 폴더 생성 및 설정 상수는 기존과 동일] ...
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_dir = os.path.join(current_dir, "view_simulation_logs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)
NUM_SCENARIOS = 200
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 150)
GRAY = (100, 100, 100)

class Object:
    def __init__(self, obj_id):
        self.original_id = obj_id
        self.id = obj_id
        
        if random.random() < 0.5:
            self.x = random.uniform(20, TARGET_X_RANGE[0] - 30)
        else:
            self.x = random.uniform(TARGET_X_RANGE[1] + 10, CANVAS_SIZE - 40)
        
        self.y = random.uniform(20, CANVAS_SIZE - 40)
        self.w = random.uniform(15, 25)
        self.h = random.uniform(15, 25)
        
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        
        self.log_data = []
        self.path = []
        self.has_entered = False

    def update(self):
        self.path.append((self.x + self.w/2, self.y + self.h/2))
        
        accel_range = 0.2
        self.vx += random.uniform(-accel_range, accel_range)
        self.vy += random.uniform(-accel_range, accel_range)
        
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 5:
            self.vx = (self.vx / speed) * 5
            self.vy = (self.vy / speed) * 5
        elif speed < 1:
            self.vx = (self.vx / (speed + 0.1)) * 1.5
            self.vy = (self.vy / (speed + 0.1)) * 1.5

        self.x += self.vx
        self.y += self.vy

        if self.x < 0 or self.x + self.w > CANVAS_SIZE:
            self.vx *= -1
            self.x = max(0, min(self.x, CANVAS_SIZE - self.w))
        if self.y < 0 or self.y + self.h > CANVAS_SIZE:
            self.vy *= -1
            self.y = max(0, min(self.y, CANVAS_SIZE - self.h))

        noise = lambda: random.uniform(0.00009, 0.009)
        self.x += noise(); self.y += noise(); self.w += noise(); self.h += noise()

        if self.id != self.original_id:
            if random.random() < 0.5: self.id = self.original_id
        else:
            if random.random() < 0.01: self.id = random.randint(1, NUM_SCENARIOS)

        if TARGET_X_RANGE[0] < self.x < TARGET_X_RANGE[1]:
            self.has_entered = True

    def draw(self, screen, font):
        if len(self.path) > 2:
            pygame.draw.lines(screen, GRAY, False, self.path, 1)
        rect = pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))
        pygame.draw.rect(screen, WHITE, rect, 1)
        id_txt = font.render(str(self.id), True, WHITE)
        screen.blit(id_txt, (int(self.x), int(self.y) - 15))

    # --- [핵심 수정 부분] ---
    def log(self):
        # 1. 좌표 변환 (x1, y1, x2, y2)
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.w, self.y + self.h
        
        # 2. confidence 생성 (0.7 ~ 0.9)
        conf = random.uniform(0.7, 0.9)
        
        # 3. C++ 구조체 순서: x1, y1, x2, y2, confidence, class_id
        log_str = f"{x1:.6f}, {y1:.6f}, {x2:.6f}, {y2:.6f}, {conf:.6f}, {self.id}"
        self.log_data.append(log_str)

# --- 메인 루프 ---
pygame.init()
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))
pygame.display.set_caption("Movement Trace Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 12)

running = True
for i in range(1, NUM_SCENARIOS + 1):
    obj = Object(i)
    filename = f"scenario_{i:03d}.txt"
    
    while not obj.has_entered and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False

        screen.fill(BLACK)
        target_rect = pygame.Rect(TARGET_X_RANGE[0], 0, TARGET_X_RANGE[1] - TARGET_X_RANGE[0], CANVAS_SIZE)
        pygame.draw.rect(screen, BLUE, target_rect)

        obj.update()
        obj.log()  # 로그 기록
        obj.draw(screen, font)

        pygame.display.flip()
        clock.tick(FPS)

    if obj.log_data:
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write("\n".join(obj.log_data))
    if not running: break

pygame.quit()