import os
import shutil

folder_path = 'test_cpp_ip/single_core_test_result_image/single_merged_image'  # 대상 폴더
preview_file = 'preview.jpg' # VS Code에서 열어둘 파일 이름
images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

for img_name in images:
    src = os.path.join(folder_path, img_name)
    shutil.copy(src, preview_file) # 파일을 preview.jpg로 복사
    
    print(f"현재 사진: {img_name}")
    user_input = input("다음 사진을 보려면 Enter, 종료하려면 q를 입력하세요: ")
    
    if user_input.lower() == 'q':
        break