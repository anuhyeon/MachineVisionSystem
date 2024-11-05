##### 1.Kernel의 크기를 3x3, 5x5, 7x7로 변경해가며 average filtering을 수행 #####
import cv2 as cv
import os
img = cv.imread('./resized_images/resized_102811.jpg')
output_dir = './blur/' # 5x5  7x7
kernel_sizes = [3, 5, 7]
# 각 커널 크기에 대해 평균 필터 적용 및 저장
for k in kernel_sizes:
    blur = cv.blur(img, ksize=(k, k))
    output_path = os.path.join(output_dir, f'filtered_{k}x{k}.jpg')
    cv.imwrite(output_path, blur)
    print(f"{output_path}에 필터링된 이미지 저장완료.")
    cv.imshow(f'Filtered Image {k}x{k}', blur)
    cv.waitKey(0)
cv.destroyAllWindows()

