import numpy as np
import cv2 as cv

image = cv.imread('/Users/an-uhyeon/MachineVisionSystem/Images/IMG_1368 2.jpg')
img = cv.resize(image,(640,480))
# cv.imwrite('./resized_images/resized_by_cv.jpg',img)
corner_img = img.copy()  # 원본 이미지 복사
corner_gray = cv.cvtColor(corner_img,cv.COLOR_BGR2GRAY) # 코너 검출을 위해 입력으로 그레이 스케일 이미지를 사용 -> 컬러이미지 사용 불가
corner_gray = np.float32(corner_gray)

# dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04) # blockSize :2x2 -> 코너 검출을 위해 고려할 주변 픽셀 영역의 크기  , ksize->커널의 크기(3x3), k=0.04 헤리스 코너 감지 알고리즘의 경험적 매개변수
# dst_ = cv.normalize(dst, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1) # 입력 데이터의 최솟값과 최대값을 기준으로 0-255 범위로 정규화,  cv.CV_8UC1: 결과 배열의 데이터 타입, 8비트 단일 채널 이미지로 변환
# img[dst>0.01*dst.max()]=[0,0,255] # 코너 표시 - dst 배열에서 최대값(dst.max())의 1% 이상인 값(즉, 코너로 판단되는 픽셀)을 찾아서 마스크로 생성. img[mask] = [0, 0, 255] - 원본 이미지(img)의 해당 위치를 빨간색([0, 0, 255] - BGR 포맷)으로 표시.
# cv.imwrite('./afterHCD/dst.jpg',dst_)

# 1) 기존 goodFeaturesToTrack 사용 (코너 표시)
corners = cv.goodFeaturesToTrack(corner_gray, maxCorners=100,qualityLevel=0.1, minDistance=10)
corners = np.intp(corners)
for c in corners:
    x, y = c.ravel()
    cv.circle(corner_img,(x,y),3,(0,0,255),-1)
cv.imwrite('./afterHCD/hcd_detected_corner_gFTT.jpg',corner_img)

# 2) 이미지를 90도 회전한 후 Harris Corner Detector 적용
rotated_img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)  # 이미지 90도 시계 방향 회전
rotated_gray = cv.cvtColor(rotated_img, cv.COLOR_BGR2GRAY)  # 회전된 이미지 그레이스케일 변환
rotated_gray = np.float32(rotated_gray)

rotated_corners = cv.goodFeaturesToTrack(rotated_gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
rotated_corners = np.intp(rotated_corners)
for c in rotated_corners:
    x, y = c.ravel()
    cv.circle(rotated_img, (x, y), 3, (0, 0, 255), -1)  # 노란색으로 goodFeaturesToTrack 코너 표시
cv.imwrite('./afterHCD/hcd_detected_corner_gFTT_rotated.jpg',rotated_img)

# 3) 영상의 크기를 2배 키운후 corner 검출
scaled_img = cv.resize(img,(640*2,480*2))
# scaled_img = cv.resize(img, None, fx=2.0, fy=2.0) #, interpolation=cv.INTER_LINEAR)

scaled_gray = cv.cvtColor(scaled_img,cv.COLOR_BGR2GRAY) # 코너 검출을 위해 입력으로 그레이 스케일 이미지를 사용 -> 컬러이미지 사용 불가
scaled_gray = np.float32(scaled_gray)

scaled_corners = cv.goodFeaturesToTrack(scaled_gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
scaled_corners = np.intp(scaled_corners)
for c in scaled_corners:
    x, y = c.ravel()
    cv.circle(scaled_img, (x, y), 3, (0, 0, 255), -1)  # 노란색으로 goodFeaturesToTrack 코너 표시
cv.imwrite('./afterHCD/hcd_detected_corner_gFTT_scaled_2.jpg',scaled_img)





# cv.imshow('Harris Corner Detection',dst_ )
#cv.waitKey(0)
# cv.imshow('img',img )
# cv.waitKey(0)
# cv.imshow('img',corner_img )
# cv.waitKey(0)
# cv.destroyAllWindows()