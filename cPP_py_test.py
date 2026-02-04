import mmap
import os
import posix_ipc
from PIL import Image
import cv2
import numpy as np
import argparse
import sys
import time
import concurrent.futures

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))
sys.path.append('/home/orangepi/Projects/rknn_model_zoo/')

from py_utils.coco_utils import COCO_test_helper

from yolo8_rknn import setup_rknn 

IMG_SIZE = (480, 480)

def run(model, img_src) :
    co_helper = COCO_test_helper(enable_letter_box=True)

    img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = img

    outputs = model.run([input_data])

    return outputs

class SharedMemoryReader:
    def __init__(self, name, width, height):
        self.width = width
        self.height = height
        self.size = width * height * 3
        self.base_name = name
        
        try:
            # 세마포어 연결
            self.sem_full = posix_ipc.Semaphore(f"/{self.base_name}_full")
            self.sem_empty = posix_ipc.Semaphore(f"/{self.base_name}_empty")
            self.sem_exit = posix_ipc.Semaphore(f"/{self.base_name}_exit")
            
            # 공유 메모리 연결 (Linux /dev/shm 기준)
            shm_path = f"/dev/shm/{self.base_name}_shm"
            self.fd = os.open(shm_path, os.O_RDONLY)
            self.shm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)
            print(f"Successfully connected to SHM: {self.base_name}")
            
        except posix_ipc.ExistentialError:
            raise Exception("C++ 프로그램을 먼저 실행해야 합니다 (세마포어를 찾을 수 없음).")

    def get_frame(self):
        """
        공유 메모리에서 프레임을 가져와 PIL Image 객체로 반환합니다.
        """
        # C++이 데이터를 쓸 때까지 대기
        self.sem_full.acquire()
        
        # 메모리 읽기 (복사가 아닌 뷰 참조로 성능 최적화)
        img = Image.frombytes("RGB", (self.width, self.height), self.shm)
        
        # 읽기 완료 신호 전송 (C++에게 다음 프레임 쓰기 허가)
        self.sem_empty.release()
        
        return img
    
    def is_exit(self) -> bool:
        try:
            self.sem_exit.acquire(0)
            print("exit")
            return True
        except posix_ipc.BusyError:
            return False

    def close(self):
        """리소스 정리"""
        if self.shm != None : 
            self.shm.close()
        os.close(self.fd)

    def __del__(self):
        try :
            self.close()
        except :
            pass

def main():
    global Model
    target = 'rk3588'
    model_path = 'model_rknn/817.rknn'
    model_1_path = 'model_rknn/822.rknn'
    model_2_path = 'model_rknn/822.rknn'

    model = setup_rknn(model_path, target, core_mask=0x1) # core 1
    model_1 = setup_rknn(model_1_path, target, core_mask=0x2) # core 2
    model_2 = setup_rknn(model_2_path, target, core_mask=0x4) # core 3

    # 1. 설정 및 초기화
    WIDTH, HEIGHT = 640, 480
    reader = SharedMemoryReader("yolo_frame", WIDTH, HEIGHT)

    print(">>> Python Inference Loop Started...")
    prev_time = 0
    frame_count = 0
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        try:
            while True:
                if reader.is_exit() :
                    break

                # 2. 공유 메모리에서 이미지 획득
                pil_img = reader.get_frame()
                #if pil_img is None:
                #    continue

                current_time = time.time()
                
                # 3. 모델 추론 함수 호출 (사용자 설계에 맞게 img 전달)
                img_input = np.array(pil_img)
                del pil_img

                future_0 = executor.submit(run, model, img_input)
                future_1 = executor.submit(run, model_1, img_input)
                future_2 = executor.submit(run, model_2, img_input)
		
                outputs = future_0.result()
                outputs_1 = future_1.result()
                outputs_2 = future_2.result()

                del img_input

                frame_count += 1

                d_t = current_time - prev_time
                if d_t > 0:
                    fps = 1 / d_t
                else:
                    fps = 0

                prev_time = current_time

                # 4. (옵션) 화면 출력용 - 디버깅 시에만 사용
                #display_frame = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
                #cv2.imshow("Shared Memory Stream", display_frame)

                print(f"\r[Inference] FPS: {fps:6.2f} | Device: {target}", end='')

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

        except KeyboardInterrupt:
            print("\nStop signal received.")
        finally:
            end = time.time()
            flow_time = end - start_time
            print(f'avg FPS : {frame_count / flow_time}')
            reader.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
