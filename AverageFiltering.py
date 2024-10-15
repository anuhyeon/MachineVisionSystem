##### 1.Kernel의 크기를 3x3, 5x5, 7x7로 변경해가며 average filtering을 수행 #####
import cv2 as cv
import os
# from google.colab.patches import cv_imshow

img = cv.imread('./resized_images/resized_102811.jpg')

# blur = cv.blur(img,ksize=(3,3))
# blur = cv.blur(img,ksize=(5,5))
# blur = cv.blur(img,ksize=(7,7))

output_dir = './blur/' # 5x5  7x7

# 커널 크기 목록
kernel_sizes = [3, 5, 7]

# 각 커널 크기에 대해 평균 필터 적용 및 저장
for k in kernel_sizes:
    # Average Filtering 적용
    blur = cv.blur(img, ksize=(k, k))
    
    # 필터링된 이미지 파일 저장 경로 설정
    output_path = os.path.join(output_dir, f'filtered_{k}x{k}.jpg')
    
    # 이미지 저장
    cv.imwrite(output_path, blur)
    print(f"{output_path}에 필터링된 이미지가 저장되었습니다.")

    # 이미지 표시
    cv.imshow(f'Filtered Image {k}x{k}', blur)
    
    # 키 입력 대기 (0이면 무한 대기)
    cv.waitKey(0)

# 모든 창 닫기
cv.destroyAllWindows()