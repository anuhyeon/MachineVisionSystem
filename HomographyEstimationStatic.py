import numpy as np
import cv2 as cv

# img_dst
# 좌표: x=262, y=186 - 좌상단
# 좌표: x=404, y=162 - 우상단
# 좌표: x=402, y=418 - 우하단
# 좌표: x=261, y=396 - 좌하단

# Read images and points
im_src = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1423.jpg')
img_src = cv.resize(im_src,(480,640))
# cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/resized_images/resized_im_src_fire.jpg',img_src)
size = img_src.shape
print(f'img_src size : {size}')
pts_src = np.array([[0,0,1],[size[1]-1,0,1],[size[1]-1,size[0]-1,1],[0,size[0]-1,1]],dtype=float)  #(좌상단, 우상단, 우하단, 좌하단) , homogeneous coordinate을 사용하기 때문에 차원을 하나 늘려주고 값은 1을 넣어줌.

im_dst = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1417.jpg')
img_dst = cv.resize(im_dst,(480,640))
# cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/resized_images/resized_im_dst_poster.jpg',img_dst)
print(f'img_dst size : {img_src.shape}')
pts_dst = np.array([[262,186,1],[404,162,1],[402,418,1],[261,396,1]],dtype=float) # homogeneous coordinate을 사용하기 때문에 차원을 하나 늘려주고 값은 1을 넣어줌.

# Calculate Homography between source and destination points
# h, status = cv.findHomography(pts_src, pts_dst)
A1 = np.hstack((np.zeros((len(pts_dst),3)),-np.expand_dims(pts_dst[:,2],axis=1)*pts_src,np.expand_dims(pts_dst[:,1],axis=1)*pts_src))
A2 = np.hstack((np.expand_dims(pts_dst[:,2],axis=1)*pts_src,np.zeros((len(pts_dst),3)),-np.expand_dims(pts_dst[:,0],axis=1)*pts_src))
A = np.vstack((A1, A2))
u, s, vh = np.linalg.svd(A, full_matrices=True)
h = vh[-1,:]/vh[-1,-1]
h = np.reshape(h,(3,3))
# h_, status = cv.findHomography(x, x_)

# Warp source image
img_temp = cv.warpPerspective(img_src, h, (img_dst.shape[1],img_dst.shape[0]))
# Black out polygonal area in destination image.
pts_dst_2d = pts_dst[:, :2]
cv.fillConvexPoly(img_dst, pts_dst_2d.astype(int), 0)
# Add warped source image to destination image.
img_dst = img_dst + img_temp
cv.imwrite('/Users/an-uhyeon/MachineVisionSystem/afterHomographyEstimation/nonusefindhomo.jpg',img_dst)
# Display image.
cv.imshow('output',img_dst)
cv.waitKey(0)
cv.destroyAllWindows()
