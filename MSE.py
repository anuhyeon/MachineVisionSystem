import cv2 as cv
import numpy as np
import glob
import os

# MSE (Mean Squared Error) 계산 함수
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def find_best_filter_size(noisy_img, clean_img, kernel_sizes):
    best_kernel = None
    min_mse = float('inf')
    for k in kernel_sizes:
        filtered_img = cv.medianBlur(noisy_img, k)
        error = mse(filtered_img, clean_img)
        print(f"kernel size: {k}, MSE: {error}")
        # 최소 MSE를 가진 필터 크기 선택
        if error < min_mse:
            min_mse = error
            best_kernel = k
    print(f"최적의 필터 크기: {best_kernel} (MSE: {min_mse})")
    return best_kernel

original_img = cv.imread('./resized_images/resized_102811.jpg')
noisy_img_pathes = glob.glob(os.path.join('./saltandpeppernoise/', '*.jpg'))#'*.jpg'
kernel_sizes = [3, 5, 7, 9, 11, 13]

for noisy_img_path in noisy_img_pathes: 
    print(noisy_img_path)
    noisy_img = cv.imread(noisy_img_path)
    best_kernel_size = find_best_filter_size(noisy_img, original_img, kernel_sizes)
    best_filtered_img = cv.medianBlur(noisy_img, best_kernel_size)
    cv.imwrite(f'./filtered_image_best_kernel/filtered_image_best_kernel_{best_kernel_size}.jpg', best_filtered_img)

