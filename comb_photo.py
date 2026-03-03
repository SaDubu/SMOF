import cv2
import os
import numpy as np

path1 = 'test_cpp_ip/test_result_image/python_image'
path2 = 'test_cpp_ip/test_result_image/cpp_image'
path3 = 'result/yolov8'
save_path = 'test_cpp_ip/test_result_image/merged_image'

if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_numbers_only(file_name):
    try:
        return int(os.path.splitext(file_name)[0])
    except ValueError:
        return 999999  

list1 = [f for f in os.listdir(path1) if f.endswith(('.jpg', '.png'))]
list2 = [f for f in os.listdir(path2) if f.endswith(('.jpg', '.png'))]
list3 = [f for f in os.listdir(path3) if f.endswith(('.jpg', '.png'))]

list1.sort(key=get_numbers_only)
list2.sort(key=get_numbers_only)
list3.sort(key=get_numbers_only)

# 2. 이번에는 리스트에서 제거하지 않습니다. 
black_out_files = ['763.jpg', '766.jpg']

# 기준을 list1(전체 시퀀스)로 잡습니다.
total_count = len(list1)
print(f"Total images to process: {total_count}")

j = 0

for i in range(total_count):
    f1 = list1[i]
    img1_path = os.path.join(path1, f1)
    img1 = cv2.imread(img1_path)
    
    if img1 is None:
        continue

    h1, w1 = img1.shape[:2]

    if f1 in black_out_files:
        img2 = np.zeros((h1, w1, 3), dtype=np.uint8)
        f2_name = "BLACK_OUT"
    else:
        f2 = list2[j]
        img2_path = os.path.join(path2, f2)
        img2 = cv2.imread(img2_path)
        f2_name = f2
        j+=1
    
    f3 = list1[i]
    img3_path = os.path.join(path3, f3)
    img3 = cv2.imread(img3_path)

    merged = cv2.hconcat([img1, img2, img3])

    save_name = f"merged_{i:04d}_{f1}"
    cv2.imwrite(os.path.join(save_path, save_name), merged)
    
    if i % 50 == 0:
        print(f"Processing... [{i}/{total_count}] (Current: {f1} + {f2_name})")

print("--- 모든 작업 완료 ---")