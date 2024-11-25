import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# import subprocess
# subprocess.call("pip install -U opencv-python".split())

img1 = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1378 2.jpg',cv.IMREAD_GRAYSCALE)
img1 = cv.resize(img1,(480,640))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/origin_img1.jpg',img1)

img2 = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1380.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.resize(img2,(640,480))
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/origin_img2.jpg',img2)

# find the keypoints and descriptors
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None) 
blob_detected_img1 = cv.drawKeypoints(img1,kp1,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 첫번째 인자는 원본 이미지여야함.

kp2, des2 = sift.detectAndCompute(img2,None)
blob_detected_img2 = cv.drawKeypoints(img2,kp2,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('Blob detected result',blob_detected_img1)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/blob_detected_img1.jpg',blob_detected_img1)
cv.imshow('Blob detected  result',blob_detected_img2)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/blob_detected_img2.jpg',blob_detected_img2)



# Match descriptors.
bf = cv.BFMatcher()
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 15 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:15], None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('matching results',img3)
cv.waitKey(0)
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/matching_result.jpg',img3)

# img1 = cv.resize(cv.rotate(img1,cv.ROTATE_90_CLOCKWISE),dsize=(0, 0),fx=1.3,fy=1.3)

plt.plot(np.arange(128), des1[matches[0].queryIdx])
plt.plot(np.arange(128), des2[matches[0].trainIdx])
plt.savefig("/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/similar_plot.jpg", dpi=300, bbox_inches='tight')  # 파일 이름, 해상도 설정, 여백 제거
plt.show()

plt.plot(np.arange(128), des1[matches[0].queryIdx])
plt.plot(np.arange(128), des2[matches[123].trainIdx])
plt.savefig("/Users/an-uhyeon/MachineVisionSystem/ImageMatchimg_SIFT/different_plot.jpg", dpi=300, bbox_inches='tight')  
plt.show()

cv.destroyAllWindows()
