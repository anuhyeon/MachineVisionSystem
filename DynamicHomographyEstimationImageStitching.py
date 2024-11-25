import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1444.jpg',cv.IMREAD_GRAYSCALE)
img1 = cv.resize(img1,(480,640))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/origin_img1.jpg',img1)

img2 = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1445.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.resize(img2,(480,640))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/origin_img2.jpg',img2)

# find the keypoints and descriptors
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None) 
blob_detected_img1 = cv.drawKeypoints(img1,kp1,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 첫번째 인자는 원본 이미지여야함.

kp2, des2 = sift.detectAndCompute(img2,None)
blob_detected_img2 = cv.drawKeypoints(img2,kp2,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Blob detected result',blob_detected_img1)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/blob_detected_img1.jpg',blob_detected_img1)

cv.imshow('Blob detected  result',blob_detected_img2)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/blob_detected_img2.jpg',blob_detected_img2)


# Match descriptors.
bf = cv.BFMatcher()
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 15 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:30], None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('matching results',img3)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/matching_result.jpg',img3)

print(len(matches))

# Read images and points - 컬러 영상으로 하기 위해서 다시 이미지 로드
im_src = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1444.jpg')
img_src = cv.resize(im_src,(480,640))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/resized_im_src.jpg',img_src)
size = img_src.shape
print(f'img_src size : {size}')

im_dst = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1445.jpg')
img_dst = cv.resize(im_dst,(480,640))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/resized_im_dst.jpg',img_dst)
print(f'img_dst size : {img_src.shape}')

# Homography 계산
if len(matches) >= 4:  # 최소 4개의 매칭이 있어야 Homography 계산 가능
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:80]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:80]]).reshape(-1, 1, 2)

    # RANSAC을 이용해 Homography 계산
    h, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    print(f"Homography matrix:\n{h}")
else:
    print("Not enough matches to compute Homography.")
    exit()

# 이미지 스티칭
# img2의 크기를 기준으로 변환된 img1을 확장
height, width = img2.shape
result = cv.warpPerspective(img1, h, (width * 2, height))  # 폭을 두 배로 확장
# 변환된 img1 오른쪽에 img2를 복사
result[0:height, 0:width] = img2

# im_out = cv.warpPerspective(img_src, h, (img_dst.shape[1] + img_src.shape[1], img_dst.shape[0]))
# im_out[0:img_src.shape[0], 0:img_src.shape[1]] = im_dst


# 결과 출력 및 저장
cv.imshow('Stitched Image', result)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/stitched_result.jpg', result)
