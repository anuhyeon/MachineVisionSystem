import cv2 as cv
import numpy as np

img = cv.imread('./Images/lanes.bmp', cv.IMREAD_GRAYSCALE)
Ix = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3) 
Iy = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
mag = np.sqrt(Ix**2 + Iy**2)
ori = np.arctan2(Iy, Ix) * 180 / np.pi

result1 = np.zeros(img.shape)
id1 = np.where(mag > 100) # 조건에 맞는 픽셀 찾기
result1[id1] = 255  # 해당 픽셀을 흰색으로 설정

# 5. Magnitude가 100 이상이고, Orientation이 30~60도 사이인 픽셀들 표시 (우측 차선)
result2 = np.zeros(img.shape)
id2 = np.where((mag > 100) & (ori > 30) & (ori <80))  #ori >30 ori < 80
result2[id2] = 255

# 6. Magnitude가 100 이상이고, Orientation이 -60~-30도 사이인 픽셀들 표시 (좌측 차선)
result3 = np.zeros(img.shape)
id3 = np.where((mag > 100) & (ori > -60) & (ori < -30)) # (ori > -80) & (ori < -30))
result3[id3] = 255

cv.imwrite('./aftersobelfilter/result1.jpg',result1)
cv.imwrite('./aftersobelfilter/result2.jpg',result2)
cv.imwrite('./aftersobelfilter/result3.jpg',result3)

# 7. 결과 출력
cv.imshow('Original Image', img)
cv.waitKey(0)
cv.imshow('result1', result1)
cv.waitKey(0)
cv.imshow('result2_left_line', result2)
cv.waitKey(0)
cv.imshow('result3_right_line', result3)
cv.waitKey(0)

cv.destroyAllWindows()


