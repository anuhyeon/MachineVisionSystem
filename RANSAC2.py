import numpy as np
import matplotlib.pyplot as plt
import cv2

def f(x,a,b):
    return a*x+b

def ransac_line_fitting(x,y,r,t):  # x,y = 데이터 포인트들의 좌표(각각 list로도 여러 값들이 들어올 수 있음), r = outlier의 비율, t = inlier를 판단할 때 쓰는 임계치
    iter = np.round(np.log(1-0.999) / np.log(1-(1-r)**2) + 1) # 0.999 = 내가 원하는 성공확률 (나는 99퍼센트 성공을 원한다!) ,iter = 99.9% 확률로 성공하기 위해 필요한 반복 횟수
    print("RANSAC 반복 : ", iter)
    num_max = 0 
    for i in np.arange(iter):
        ################ 랜덤으로 2개의 점을 선택 ################
        id = np.random.permutation(len(x)) # permutation 랜덤하게 len(x)개수의 점들을 섞음
        #print("_+_+_+_+_+_+_+_+",id[0],id[1],len(id),len(x))
        xs = x[id[:2]] # 2개의 점 x좌표
        ys = y[id[:2]] # 2개의 점 y좌표
        ####################################################
        
        ###### 두점을 이용해서 직선을 구함 기울기a와 절편b를 계산 ######
        A = np.vstack([xs, np.ones(len(xs))]).T # 행렬 A 생성 (x좌표와 상수 1로 이루어진 행렬)
        ab = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ys)) # 기울기와 절편 계산
        ####################################################
        
        dist = np.abs(ab[0]*x-y+ab[1])/np.sqrt(ab[0]**2+1) # 모든 점들과 위에서 구한 직선 사이의 거리를 구함.
        numInliers = sum(dist < t) # 설정한 임계치 이하의 거리를 가진 친구들의 개수
        if numInliers > num_max: # 계속 갱신
            ab_max = ab
            num_max = numInliers
    return ab_max, num_max


# 좌측 차선 픽셀들이 표시된 영상 로드
line_image = cv2.imread('./aftersobelfilter/result3.jpg', cv2.IMREAD_GRAYSCALE)  # 좌측 차선 픽셀들이 흰색(255)로 표시된 영상
original_image = cv2.imread('./Images/lanes.bmp')
right_image = cv2.imread('./aftersobelfilter/result3.jpg')

# 흰색 픽셀의 x, y 좌표 추출
idd = np.where(line_image == 255)  # 픽셀 값이 255인 좌표만 추출 -> np.where는 y좌표 리스트와 x좌표 리스트 2개를 반환
xno = idd[1]
yno = idd[0]
print(xno,yno)
abno, max_inliers = ransac_line_fitting(xno,yno,0.5,2)

print("기울기와 절편:", abno)

# 원본 이미지에 직선을 그리기 위한 시작점과 끝점 계산
m, b = abno
print("original image shape : ",original_image.shape)

height, width = original_image.shape[:2]
x_start, x_end = 0, width  # x 좌표의 시작과 끝을 이미지 너비로 설정
y_start, y_end = int(f(x_start, m, b)), int(f(x_end, m, b))  # f(x, a, b)로 y 좌표 계산 ->  int(f(x_start, a, b))처럼 int로 형 변환을 하는 이유는 이미지에서 좌표는 반드시 정수 값이어야 하기 때문! - 특히 cv2.line()과 같은 함수들은 좌표 값으로 정수를 요구함.
print("x좌표 시작과 끝 : ", x_start, x_end)
print("y좌표 시작과 끝 : " ,y_start, y_end)
# 직선을 원본 이미지에 그리기
output_image = original_image.copy()
# output_image = right_image.copy()

cv2.line(output_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)  # 파란색으로 직선 그리기

# 결과 출력
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Original Image with Fitted Line Lane)")
plt.show()

cv2.imwrite('./aftersobelfilter/left_lane_fitting.jpg',output_image)
