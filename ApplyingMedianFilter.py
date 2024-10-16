import cv2 as cv 
import numpy as np
import glob
import os

noisy_images = glob.glob(os.path.join('./saltandpeppernoise/', '*.jpg'))#'*.jpg'
# 각 Noise 비율에 대해 Median Filter 적용
kernel_sizes = [3, 5, 7, 9, 11,13]  # 테스트할 커널 사이즈
# 세 가지 비율로 Salt and Pepper Noise 추가
ratios = [0.02, 0.25, 0.1]
# print(list(zip(ratios,noisy_images)))
for ratio, noisy_img_path in zip(ratios, noisy_images):
    print(ratio, noisy_img_path)
    noisy_img = cv.imread(noisy_img_path)
    for kernel_size in kernel_sizes:
        median_img = cv.medianBlur(noisy_img, kernel_size)
        cv.imwrite(f'./aftermedianfilter/median_filtered_ratio_{ratio}_kernel_{kernel_size}.jpg', median_img)
        print(f"Saved: ./aftermedianfilter/median_filtered_ratio_{ratio}_kernel_{kernel_size}.jpg")

print("Median Filtering이 적용된 이미지가 생성되었습니다.")
