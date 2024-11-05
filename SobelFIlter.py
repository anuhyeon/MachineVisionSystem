import cv2 as cv
import numpy as np

image = cv.imread('./images/lanes.bmp',cv.IMREAD_GRAYSCALE)
# 1. x방향 미분 계산 (dx=1, dy=0)
grad_x = cv.Sobel(image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)  # x 방향 미분
# 2. y방향 미분 계산 (dx=0, dy=1)
grad_y = cv.Sobel(image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)  # y 방향 미분
# 3. Gradient Magnitude 계산
magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y)) # 크기 계산
# 4. Gradient Orientation 계산
orientation = np.arctan2(grad_y, grad_x)*180/np.pi  # 방향 계산 (라디안 -> degree단위)

# 5. 결과 이미지 정규화 (0~255 사이로 변환)
grad_x_ = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min())*255
grad_y_ = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min())*255
magnitude_ = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())*255
orientation_ = (orientation - orientation.min()) / (orientation.max() - orientation.min())*255

grad_x_ = grad_x_.astype(np.uint8)
grad_y_ = grad_y_.astype(np.uint8)
magnitude_ = magnitude_.astype(np.uint8)
orientation_ = orientation_.astype(np.uint8)

cv.imwrite('./aftersobelfilter/grad_x.jpg',grad_x_)
cv.imwrite('./aftersobelfilter/grad_y.jpg',grad_y_)
cv.imwrite('./aftersobelfilter/magnitude.jpg',magnitude_)
cv.imwrite('./aftersobelfilter/orientation.jpg',orientation_)


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


