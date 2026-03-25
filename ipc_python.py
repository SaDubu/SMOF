import mmap
import os
import posix_ipc
from PIL import Image
import cv2
import numpy as np
import sys
import time
import concurrent.futures
import struct

from py_utils.coco_utils import COCO_test_helper

from yolo8_rknn import setup_rknn, sigmoid_post_process, post_process, pack_add_draw 

IMG_SIZE = (640, 640)
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

        self.sequence_num = 1
        
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

            shm_path_chips = f"//dev/shm/{self.base_name}_chips_num"
            self.fd_chips = os.open(shm_path_chips, os.O_RDWR)
            self.mem_chips = mmap.mmap(self.fd_chips, 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)

            print(f"Successfully connected to SHM: {self.base_name}")
            
        except posix_ipc.ExistentialError:
            raise Exception("C++ 프로그램을 먼저 실행해야 합니다.")

    def get_frame(self):
        if self.is_exit() :
            return None
        self.sem_full.acquire()
        
        raw = np.frombuffer(self.shm, dtype=np.uint8).reshape((self.height, self.width, 3))
        img = raw.copy()
        self.sem_empty.release()
        
        return img
    
    def set_chips_num(self, num):
        binary_data = struct.pack('i', num)
        
        self.mem_chips[:4] = binary_data

    def get_chips_num(self):
        binary_data = self.mem_chips[:4]
        
        num = struct.unpack('i', binary_data)[0]
        
        return num
        
    
    def send_yolo_result(self, detections):
        try:
            count = len(detections) if detections is not None and len(detections) > 0 else 0
            count = min(count, self.max_objects)

            # Hxx: Unsigned Short(2B) + Padding(2B) = 총 4바이트 헤더
            header = struct.pack('Hxx', self.sequence_num)
            count_bytes = struct.pack('i', count)

            data_to_send = detections[:count].astype(np.float32).tobytes() if count > 0 else b""
            #print(detections[0])

            self.sem_yolo.acquire()
            try:
                self.mem_yolo.seek(0)
                self.mem_yolo.write(header)       # 0~3 처리한 frame과 매칭을 위한 부분
                self.mem_yolo.write(count_bytes)  # 4~7 현 frame에서 감지된 object의 수
                if count > 0:
                    self.mem_yolo.write(data_to_send) # 8~ boxes score class 순서로 쭉 담아서 보냄.
                
                self.sequence_num = 1 if self.sequence_num >= 65535 else self.sequence_num + 1
            finally:
                self.sem_yolo.release()

        except Exception as e:
            print(f"Error in send_yolo_result: {e}")
    def is_exit(self) -> bool:
        try:
            self.sem_exit.acquire(0)
            print("\nexit")
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

def n_post_process(output_data, co_helper) :
    boxes, classes, scores = sigmoid_post_process(output_data)
    #boxes, classes, scores = post_process(output_data)

    if boxes is not None :
        output = np.column_stack((boxes, scores, classes))

        return output, boxes, scores, classes 

    return None, None, None, None  

def one_thing_left(input_data) :
    combined = np.vstack(input_data)

    max_id = np.argmax(combined[:, 4])

    left_thing = combined[max_id : max_id + 1]

    return left_thing 

def main():
    global Model
    target = 'rk3588'
    model_path = 'model_rknn/640_2.rknn'
    model_1_path = 'model_rknn/102_class.rknn'

    model = setup_rknn(model_path, target, core_mask=0x0) # core 1
    #model_1 = setup_rknn(model_1_path, target, core_mask=0x2) # core 2

    WIDTH, HEIGHT = 640, 480
    s_m_m = SharedMemory("yolo_frame", WIDTH, HEIGHT)

    print(">>> Python Inference Loop Started...")
    prev_time = 0
    frame_count = 0
    start_time = time.time()

    boxes, scores, classes = np.zeros(0), np.zeros(0), np.zeros(0)
    co_helper = COCO_test_helper(enable_letter_box=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                if s_m_m.is_exit() :
                    break

                image_num = s_m_m.get_chips_num()

                if image_num is None or image_num != 0 :
                    continue

                outputs = []
                send_data = None

                count = 1

                image_num = 1
                current_time = time.time()
                for i in range(image_num) :
                    img_input = s_m_m.get_frame()
                    if img_input is None:
                        continue

                    input_data = co_helper.letter_box(im= img_input.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))

                    output = model.run([input_data])
                    output, boxes, scores, classes = n_post_process(output, co_helper)

                    if output is None :
                        output = np.zeros(6, dtype=np.float32)
                        
                    #else :
                    #    output = one_thing_left(output) 

                    outputs.append(output)
                
                send_data = np.vstack(outputs).astype(np.float32)
                frame_count += 1
                if boxes is not None:
                    for i in classes :
                        print(f'{i}_detect')
                #    img_input = pack_add_draw(img_input, co_helper.get_real_box(boxes), scores, classes)
                #img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)

                print(f'{frame_count}...')

                #cv2.imshow("1", img_input)
                #cv2.waitKey(1)
                    
                #cv2.imwrite(f'real_time/{frame_count}.jpg', img_input)

                s_m_m.send_yolo_result(send_data)

                s_m_m.set_chips_num(0)


                # 공유 메모리에서 이미지 획득
                #img_input = s_m_m.get_frame()
                #if img_input is None:
                #    continue

                #future_0 = executor.submit(run, model, img_input)
                #future_1 = executor.submit(run, model_1, img_input)
		
                #outputs = future_0.result()

                #img = co_helper.letter_box(im= img_input.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #cv2.imwrite(f'test_cpp_ip/cpp_trans_py_image/{frame_count}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                #input_data = img

                #output = model.run([input_data])
                #outputs, boxes, scores, classes = n_post_process(output, co_helper)
                #outputs_1 = future_1.result()
                #outputs_1, boxes, scores, classes = n_post_process(outputs_1)
                #outputs_1 = None

                #if outputs is not None :
                #    outputs = one_thing_left(outputs);
                #    s_m_m.send_yolo_result(outputs)
                #else :
                #    empty_thing = np.empty((0, 6), dtype=np.float32)
                #    s_m_m.send_yolo_result(empty_thing)

                # valid_outputs = []
                # if outputs is not None:
                #     valid_outputs.append(outputs)
                # if outputs_1 is not None:
                #     valid_outputs.append(outputs_1)

                # if len(valid_outputs) > 0:
                #     out_thing = one_thing_left(valid_outputs)
                #     s_m_m.send_yolo_result(out_thing)
                # else :
                #     empty_thing = np.empty((0, 6), dtype=np.float32)
                #     s_m_m.send_yolo_result(empty_thing)

                #if boxes is not None:
                #   img_input = pack_add_draw(img_input, boxes, scores, classes)
                #   img_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
                
                #    cv2.imwrite(f'test_cpp_ip/real_time/{frame_count}.jpg', img_input)

                #del img_input

                #d_t = current_time - prev_time
                #if d_t > 0:
                #    fps = 1 / d_t
                #else:
                #    fps = 0

                #prev_time = current_time

                #print(f"\r[Inference] FPS: {fps:6.2f} | Device: {target}", end='')



                # display_debug
                #cv2.imshow("1", img_input)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #   break

        except KeyboardInterrupt:
            print("\nStop.")
        finally:
            end = time.time()
            flow_time = end - start_time
            print(f'avg FPS : {frame_count / flow_time}')
            s_m_m.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
