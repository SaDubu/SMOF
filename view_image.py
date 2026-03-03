import os
import shutil

#folder_path = 'test_cpp_ip/try_2/single_merged_image'
folder_path = 'test_cpp_ip/real_time'  
preview_file = 'preview.jpg'

images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

try:
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
except ValueError:
    images.sort()

print(f">>> 총 {len(images)}개의 이미지를 발견했습니다.")

for img_name in images:
    src = os.path.join(folder_path, img_name)
    
    shutil.copy(src, preview_file) 
    
    print(f"\r현재 사진: {img_name} (Enter: 다음 / q: 종료)", end="")
    
    user_input = input()
    
    if user_input.lower() == 'q':
        print("\n종료합니다.")
        break