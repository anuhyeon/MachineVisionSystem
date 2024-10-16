import cv2 as cv
import numpy as np
import glob
import os

# MSE (Mean Squared Error) 계산 함수
def mse(imageA, imageB):
    # 두 이미지 간의 차이 제곱합을 평균한 값 반환
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# 적절한 필터 크기를 찾는 함수
def find_best_filter_size(noisy_img, clean_img, kernel_sizes):
    best_kernel = None
    min_mse = float('inf')
    for k in kernel_sizes:
        # Median Filtering 적용
        filtered_img = cv.medianBlur(noisy_img, k)
        # 필터링된 이미지와 깨끗한 이미지 간의 MSE 계산
        error = mse(filtered_img, clean_img)
        print(f"Kernel Size: {k}, MSE: {error}")
        # 최소 MSE를 가진 필터 크기 선택
        if error < min_mse:
            min_mse = error
            best_kernel = k
    print(f"최적의 필터 크기: {best_kernel} (MSE: {min_mse})")
    return best_kernel

# 이미지 읽기 
original_img = cv.imread('./resized_images/resized_102811.jpg')
noisy_img_pathes = glob.glob(os.path.join('./saltandpeppernoise/', '*.jpg'))#'*.jpg'
# 테스트할 커널 크기 목록
kernel_sizes = [3, 5, 7, 9, 11, 13]

for noisy_img_path in noisy_img_pathes: 
    print(noisy_img_path)
    noisy_img = cv.imread(noisy_img_path)
    # 최적의 필터 크기 찾기
    best_kernel_size = find_best_filter_size(noisy_img, original_img, kernel_sizes)
    # 최적의 필터 크기로 필터링 수행 및 결과 저장
    best_filtered_img = cv.medianBlur(noisy_img, best_kernel_size)
    cv.imwrite(f'./filtered_image_best_kernel/filtered_image_best_kernel_{best_kernel_size}.jpg', best_filtered_img)
