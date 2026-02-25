import mmap
import os
import posix_ipc
from PIL import Image
import cv2
import numpy as np
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

from yolo8_rknn import setup_rknn, weck_post_process, add_post_process, post_process 

IMG_SIZE = (480, 480)
Max_OBJECTS = 100

def run(model, img_src) :
    co_helper = COCO_test_helper(enable_letter_box=True)

    img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = img

    outputs = model.run([input_data])

    return outputs

class SharedMemory:
    def __init__(self, name, width, height):
        self.width = width
        self.height = height
        self.size = width * height * 3
        self.base_name = name

        self.max_objects = Max_OBJECTS
        self.yolo_size = 4 + (self.max_objects * 6 * 4)
        
        try:
            # 세마포어 연결
            self.sem_full = posix_ipc.Semaphore(f"/{self.base_name}_full")
            self.sem_empty = posix_ipc.Semaphore(f"/{self.base_name}_empty")
            self.sem_exit = posix_ipc.Semaphore(f"/{self.base_name}_exit")
            
            # 공유 메모리 연결 (Linux /dev/shm)
            shm_path = f"/dev/shm/{self.base_name}_shm"
            self.fd = os.open(shm_path, os.O_RDONLY)
            self.shm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)

            shm_path_yolo = f"/dev/shm/{self.base_name}_yolo"
            self.fd_yolo = os.open(shm_path_yolo, os.O_RDWR)
            self.mem_yolo = mmap.mmap(self.fd_yolo, self.yolo_size, access=mmap.ACCESS_WRITE)
            self.sem_yolo = posix_ipc.Semaphore(f"/{self.base_name}_yolo_sem")

            print(f"Successfully connected to SHM: {self.base_name}")
            
        except posix_ipc.ExistentialError:
            raise Exception("C++ 프로그램을 먼저 실행해야 합니다.")

    def get_frame(self):
        self.sem_full.acquire()
        
        img = Image.frombytes("RGB", (self.width, self.height), self.shm)
        
        self.sem_empty.release()
        
        return img
    
    def send_yolo_result(self, detections):
        try:
            count = len(detections) if detections is not None and len(detections) > 0 else 0

            count = min(count, self.max_objects)

            count_bytes = np.int32(count).tobytes()

            data_to_send = detections[:count].astype(np.float32).tobytes() if count > 0 else b""

            #print(f"Current Sem Value: {self.sem_yolo.value}")
            self.sem_yolo.acquire()
            try:
                self.mem_yolo.seek(0)
                self.mem_yolo.write(count_bytes)
                if count > 0:
                    self.mem_yolo.write(data_to_send)
            finally:
                self.sem_yolo.release()

        except Exception as e:
            print(f"Error in send_yolo_result: {e}")
    def is_exit(self) -> bool:
        try:
            self.sem_exit.acquire(0)
            print("exit")
            return True
        except posix_ipc.BusyError:
            return False

    def close(self):
        if self.shm != None : 
            self.shm.close()
        if self.sem_yolo != None :
            self.sem_yolo.close()
        os.close(self.fd)

    def __del__(self):
        try :
            self.close()
        except :
            pass

#이 부분에서 병목이 생기는 것을 확인함.
#기존에 사용했던 add_post_process에서 뭔가 특징이 있는 곳만 순회하여 확인하는 방식으로 cost를 줄임.
def n_post_process(output_data) :
    #boxes, classes, scores = weck_post_process(output_data)
    boxes, classes, scores = post_process(output_data)

    if boxes is not None :
        output = np.column_stack((boxes, scores, classes))

        return output 

    return None   

def main():
    global Model
    target = 'rk3588'
    model_path = 'model_rknn/new_top.rknn'
    model_1_path = 'model_rknn/new_bottom.rknn'

    model = setup_rknn(model_path, target, core_mask=0x1) # core 1
    model_1 = setup_rknn(model_1_path, target, core_mask=0x2) # core 2

    WIDTH, HEIGHT = 480, 480
    s_m_m = SharedMemory("yolo_frame", WIDTH, HEIGHT)

    print(">>> Python Inference Loop Started...")
    prev_time = 0
    frame_count = 0
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                if s_m_m.is_exit() :
                    break

                # 공유 메모리에서 이미지 획득
                pil_img = s_m_m.get_frame()
                if pil_img is None:
                    continue

                current_time = time.time()
                
                img_input = np.array(pil_img)
                del pil_img

                future_0 = executor.submit(run, model, img_input)
                future_1 = executor.submit(run, model_1, img_input)
		
                outputs = future_0.result()
                outputs = n_post_process(outputs)
                outputs_1 = future_1.result()

                s_m_m.send_yolo_result(outputs)

                #del img_input

                frame_count += 1

                d_t = current_time - prev_time
                if d_t > 0:
                    fps = 1 / d_t
                else:
                    fps = 0

                prev_time = current_time

                print(f"\r[Inference] FPS: {fps:6.2f} | Device: {target}", end='')

                # display_debug
                img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
                cv2.imshow("Shared Memory Stream", img_input)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStop signal received.")
        finally:
            end = time.time()
            flow_time = end - start_time
            print(f'avg FPS : {frame_count / flow_time}')
            s_m_m.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
