import cv2 as cv
import numpy as np

# 1. 이미지 불러오기 (그레이스케일)
image = cv.imread('./images/lanes.bmp', cv.IMREAD_GRAYSCALE)
if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# 2. x방향 미분 계산 (dx=1, dy=0)
grad_x = cv.Sobel(image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)

# 3. y방향 미분 계산 (dx=0, dy=1)
grad_y = cv.Sobel(image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)

# 4. Gradient Magnitude 계산
magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

# 5. Gradient Orientation 계산 (라디안 -> degree 단위)
orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi

# 6. 정규화 (0~255 사이로 변환 및 uint8 타입으로 변경)
grad_x_ = cv.normalize(grad_x, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
grad_y_ = cv.normalize(grad_y, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
magnitude_ = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Orientation은 0~180 범위로 조정 후 정규화
orientation_ = ((orientation + 180) % 360) / 360 * 255
orientation_ = orientation_.astype(np.uint8)

# 7. 결과 출력
cv.imshow('Original Image', image)
cv.waitKey(0)
cv.imshow('Gradient X', grad_x_)
cv.waitKey(0)
cv.imshow('Gradient Y', grad_y_)
cv.waitKey(0)
cv.imshow('Gradient Magnitude', magnitude_)
cv.waitKey(0)
cv.imshow('Gradient Orientation', orientation_)
cv.waitKey(0)

cv.destroyAllWindows()
