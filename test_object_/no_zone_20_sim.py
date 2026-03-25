import pygame
import random
import os
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "scenario_new_logs_999")
if not os.path.exists(output_dir): os.makedirs(output_dir)

CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)
NUM_SCENARIOS = 999
MAX_FRAMES = 100000
FPS = 60 

class Object:
    def __init__(self, obj_id, is_main=False, will_enter=True):
        self.original_id = obj_id
        self.id = obj_id
        self.is_main = is_main
        self.will_enter = will_enter
        self.is_out = False 

        if is_main:
            self.side = random.choice(['left', 'right'])
            if self.side == 'left':
                self.x = random.uniform(20, 80)
                self.vx = random.uniform(2, 4)
            else:
                self.x = random.uniform(400, 450)
                self.vx = random.uniform(-4, -2) 
        else:
            self.x = random.uniform(0, CANVAS_SIZE - 20)
            self.vx = random.uniform(-3, 3)
            
        self.y = random.uniform(50, CANVAS_SIZE - 50)
        self.w, self.h = random.uniform(18, 22), random.uniform(18, 22)
        self.vy = random.uniform(-2, 2)
        self.has_entered = False

    def update(self):
        self.vx += random.uniform(-0.1, 0.1)
        self.vy += random.uniform(-0.1, 0.1)
        
        next_x = self.x + self.vx
        next_y = self.y + self.vy

        if self.is_main:
            if not self.will_enter:
                center_x = next_x + self.w / 2
                if TARGET_X_RANGE[0] < center_x < TARGET_X_RANGE[1]:
                    self.vx *= -1.2 
                    next_x = self.x 
                
                if next_x + self.w < -50 or next_x > CANVAS_SIZE + 50:
                    self.is_out = True
            else:
                if TARGET_X_RANGE[0] < (next_x + self.w/2) < TARGET_X_RANGE[1]:
                    self.has_entered = True
        
        self.x, self.y = next_x, next_y

        if not self.is_main:
            if self.x < 0 or self.x + self.w > CANVAS_SIZE: self.vx *= -1
        
        if self.y < 0 or self.y + self.h > CANVAS_SIZE: self.vy *= -1

    def get_log_str(self):
        conf = random.uniform(0.75, 0.95)
        return f"{self.x:.4f}, {self.y:.4f}, {self.x+self.w:.4f}, {self.y+self.h:.4f}, {conf:.4f}, {self.id}"

def get_motion_region(main_obj):
    offset_limit = main_obj.w * 0.15
    m_w = main_obj.w * random.uniform(0.95, 1.05)
    m_h = main_obj.h * random.uniform(0.95, 1.05)
    m_x = main_obj.x + random.uniform(-offset_limit, offset_limit)
    m_y = main_obj.y + random.uniform(-offset_limit, offset_limit)
    return f"{m_x:.4f}, {m_y:.4f}, {m_x+m_w:.4f}, {m_y+m_h:.4f}"

# --- 실행 부분 ---
pygame.init()
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))

for i in range(1, NUM_SCENARIOS + 1):
    will_enter = random.random() >= 0.2 
    main_obj = Object(i, is_main=True, will_enter=will_enter)
    noise_objs = [Object(random.randint(1000, 9999)) for _ in range(random.randint(2, 4))]
    
    scenario_logs = []
    frame_count = 0
    post_out_frames = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if will_enter and main_obj.has_entered: break
        if frame_count >= MAX_FRAMES: break
        
        screen.fill((0, 0, 0))
        zone_color = (0, 0, 100) if will_enter else (80, 0, 0)
        pygame.draw.rect(screen, zone_color, (TARGET_X_RANGE[0], 0, 120, CANVAS_SIZE))

        if not will_enter and main_obj.is_out:
            post_out_frames += 1
            if post_out_frames >= FPS * 2: 
                break
            motion_str = "None" 
        else:
            main_obj.update()
            for n_obj in noise_objs:
                n_obj.update()
            motion_str = get_motion_region(main_obj)

        frame_header = f"FRAME: {frame_count}, MOTION: {motion_str}, OBJS:"
        obj_logs = [main_obj.get_log_str()]
        for n_obj in noise_objs:
            obj_logs.append(n_obj.get_log_str())
        scenario_logs.append(f"{frame_header} | {' / '.join(obj_logs)}")
        
        main_color = (255, 255, 255) if will_enter else (255, 200, 0)
        pygame.draw.rect(screen, main_color, (main_obj.x, main_obj.y, main_obj.w, main_obj.h), 2)
        for n_obj in noise_objs:
            pygame.draw.rect(screen, (70, 70, 70), (n_obj.x, n_obj.y, n_obj.w, n_obj.h), 1)

        pygame.display.flip()
        frame_count += 1
        
    filename = f"scenario_{i:03d}.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(scenario_logs))
    
    scenario_logs.clear()
    if i % 50 == 0:
        print(f"Progress: {i}/{NUM_SCENARIOS}")
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))

pygame.quit()