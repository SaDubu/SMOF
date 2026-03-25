import numpy as np
import os

# --- 1. 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "scenario_new_logs_999")
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)
    print(f"✅ 폴더 생성 완료: {output_dir}")

# --- 2. 설정값 ---
CANVAS_SIZE = 480
TARGET_X_RANGE = (180, 300)
NUM_SCENARIOS = 999
MAX_FRAMES = 3000 
FPS = 60

def get_motion_region(x, y, w, h):
    """메인 객체 주변에 약간의 오차를 둔 MOTION 영역 생성"""
    m_w = w * np.random.uniform(0.9, 1.1)
    m_h = h * np.random.uniform(0.9, 1.1)
    m_x = x + np.random.uniform(-w*0.1, w*0.1)
    m_y = y + np.random.uniform(-h*0.1, h*0.1)
    return f"{m_x:.4f} {m_y:.4f} {m_x+m_w:.4f} {m_y+m_h:.4f}"

for i in range(1, NUM_SCENARIOS + 1):
    will_enter = np.random.rand() >= 0.2  

    mw, mh = 20, 20
    side = np.random.choice(['left', 'right'])
    
    if side == 'left':
        mx = np.random.uniform(10, 50)
        vx = np.random.uniform(2.5, 4.0)
    else:
        mx = np.random.uniform(430, 470)
        vx = np.random.uniform(-4.0, -2.5)
        
    my = np.random.uniform(50, 430)
    vy = np.random.uniform(-1.5, 1.5)
    
    noises = []
    for _ in range(np.random.randint(2, 5)):
        noises.append({
            'x': np.random.uniform(50, 430), 'y': np.random.uniform(50, 430),
            'vx': np.random.uniform(-2, 2), 'vy': np.random.uniform(-2, 2),
            'id': np.random.randint(1000, 9999)
        })

    log_data = []
    frame_count = 0
    is_out = False
    post_out_frames = 0

    while frame_count < MAX_FRAMES:
        cx = mx + mw / 2
        cy = my + mh / 2

        if is_out:
            post_out_frames += 1
            if post_out_frames >= FPS * 2:
                break
        
        if not is_out:
            next_mx = mx + vx
            next_my = my + vy
            next_cx = next_mx + mw / 2
            next_cy = next_my + mh / 2

            if next_cx < 0 or next_cx > CANVAS_SIZE or \
               next_cy < 0 or next_cy > CANVAS_SIZE:
                is_out = True
                mx, my = next_mx, next_my 
            else:
                if TARGET_X_RANGE[0] <= next_cx <= TARGET_X_RANGE[1]:
                    if will_enter:
                        m_str = get_motion_region(next_mx, next_my, mw, mh)
                        objs_line = f"{next_mx:.4f} {next_my:.4f} {next_mx+mw:.4f} {next_my+mh:.4f} 0.9900 {i}"
                        log_data.append(f"FRAME: {frame_count} MOTION: {m_str} OBJS: | {objs_line}")
                        break
                    else:

                        vx *= -1.2
                        mx = mx + vx 
                else:
                    mx, my = next_mx, next_my

        for n in noises:
            n['x'] += n['vx']; n['y'] += n['vy']
            if n['x'] < 0 or n['x'] > CANVAS_SIZE - 20: n['vx'] *= -1
            if n['y'] < 0 or n['y'] > CANVAS_SIZE - 20: n['vy'] *= -1

        m_str = "None" if is_out else get_motion_region(mx, my, mw, mh)
        
        main_obj_log = f"{mx:.4f} {my:.4f} {mx+mw:.4f} {my+mh:.4f} {np.random.uniform(0.85, 0.96):.4f} {i}"
        obj_list = [main_obj_log]
        for n in noises:
            obj_list.append(f"{n['x']:.4f} {n['y']:.4f} {n['x']+20:.4f} {n['y']+20:.4f} {np.random.uniform(0.7, 0.8):.4f} {n['id']}")

        log_line = f"FRAME: {frame_count} MOTION: {m_str} OBJS: | {' / '.join(obj_list)}"
        log_data.append(log_line)
        frame_count += 1

    file_name = f"scenario_{i:03d}.txt"
    with open(os.path.join(output_dir, file_name), "w") as f:
        f.write("\n".join(log_data))

    if i % 100 == 0:
        print(f"🚀 Progress: {i}/{NUM_SCENARIOS} scenarios completed.")

print(f"✅ 완료! 모든 로그가 '{output_dir}'에 저장되었습니다.")