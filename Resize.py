import cv2
import os
import glob

# 입력 및 출력 디렉토리 설정
input_dir = './images/'  # 원본 이미지들이 있는 디렉토리
output_dir = './resized_images/'  # 리사이즈된 이미지가 저장될 디렉토리

# # 출력 디렉토리가 없으면 생성
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# 입력 디렉토리 내 모든 JPG 파일 탐색
jpg_files = glob.glob(os.path.join(input_dir, '*.jpg'))#'*.jpg'

# 파일 처리
for idx, file in enumerate(jpg_files):
    # 이미지 읽기
    image = cv2.imread(file)

    # 이미지가 정상적으로 불러와졌는지 확인
    if image is None:
        print(f"{file} 파일을 불러올 수 없습니다.")
        continue

    # 이미지 리사이즈
    resized_image = cv2.resize(image, (512, 512))
    
    # 사용자 정의 저장 이름 설정 (예: resized_1.jpg, resized_2.jpg ...)
    save_name = f"resized_1028{idx + 1}.jpg"  # 여기서 이름을 조정할 수 있습니다.

    # 출력 파일 경로 설정
    output_path = os.path.join(output_dir,save_name)

    # 리사이즈된 이미지 저장
    cv2.imwrite(output_path, resized_image)
    print(f"{output_path}에 이미지가 저장되었습니다.")

print("모든 이미지의 리사이즈가 완료되었습니다.")
