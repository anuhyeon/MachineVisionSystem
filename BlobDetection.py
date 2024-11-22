import numpy as np
import cv2 as cv
# import subprocess # 파이썬 코드에서 외부 프로그램을 실행하거나 시스템 명령어를 실행할 수있도록 지원하는 모듈
# subprocess.call("pip install -U opencv-python".split())

image = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1368 2.jpg')
img = cv.resize(image,(640,480))

# 1)
blob_img = img.copy()  # 원본 이미지 복사
gray= cv.cvtColor(blob_img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
blob_detected_img = cv.drawKeypoints(gray,kp,blob_img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('./afterBLOB/blob_detected_corner.jpg',blob_detected_img)

# 2)
rotated_blob_img = img.copy()  # 원본 이미지 복사
rotated_img = cv.rotate(rotated_blob_img, cv.ROTATE_90_CLOCKWISE)  # 이미지 90도 시계 방향 회전
rotated_gray= cv.cvtColor(rotated_img,cv.COLOR_BGR2GRAY)
rotated_sift = cv.SIFT_create()
rotated_kp = rotated_sift.detect(rotated_gray,None)
rotated_blob_detected_img = cv.drawKeypoints(rotated_gray,rotated_kp,rotated_blob_img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('./afterBLOB/blob_detected_corner_rotated.jpg',rotated_blob_detected_img)

# 3)
scaled_blob_img = img.copy()  # 원본 이미지 복사
scaled_img = cv.resize(scaled_blob_img,(640*2,480*2))
scaled_gray= cv.cvtColor(scaled_img,cv.COLOR_BGR2GRAY)
scaled_sift = cv.SIFT_create()
scaled_kp = scaled_sift.detect(scaled_gray,None)
scaled_blob_detected_img = cv.drawKeypoints(scaled_gray,scaled_kp,scaled_blob_img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('./afterBLOB/blob_detected_corner_scaled.jpg',scaled_blob_detected_img)



