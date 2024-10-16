import cv2 as cv 
import numpy as np
import time

# 1) 1D Gaussian 필터 생성 (길이=7, σ=2)
kernel1d = cv.getGaussianKernel(ksize=7, sigma=2)
print(kernel1d.shape)
# 2) 1D Gaussian 필터의 outerproduct를 사용해 2D Gaussian 필터 생성
kernel2d = np.outer(kernel1d, kernel1d.transpose())
print(kernel2d.shape)
# 3) 입력 영상을 kernel2d를 사용하여 filtering - 
img = cv.imread('./resized_images/resized_102811.jpg')
# 데이터 타입 확인
dtype = img.dtype
print(f"데이터 타입: {dtype}")

# 채널 수와 비트 깊이 계산
channels = img.shape[2] if len(img.shape) == 3 else 1
bit_depth = img.itemsize * 8  # 각 채널의 비트 깊이 

print(f"채널 수: {channels}")
print(f"비트 깊이 (채널당): {bit_depth} bits")
print(f"전체 비트 깊이 (모든 채널 합): {bit_depth * channels} bits")
print('원본이미지 사이즈:',img.shape)
print('- - - - - - - - - - - - - - - - - - - - - - -')

# float32로 변환했음 - uint8로 할 경우 opencv내부 적으로 uint8을 float32로 변환하는 과정에서 발생하는 소수점 반올림 오차 때문에 2D 필터와 separable 필터 적용후 이미지 결과간 오차가 발생할 수 있기 때문!
img = img.astype(np.float32)
# 데이터 타입 확인
dtype = img.dtype
print(f"데이터 타입: {dtype}")
# 채널 수와 비트 깊이 계산
channels = img.shape[2] if len(img.shape) == 3 else 1
bit_depth = img.itemsize * 8  # 각 채널의 비트 깊이 

print(f"채널 수: {channels}")
print(f"비트 깊이 (채널당): {bit_depth} bits")
print(f"전체 비트 깊이 (모든 채널 합): {bit_depth * channels} bits")
print('원본이미지 사이즈:',img.shape)
print('- - - - - - - - - - - - - - - - - - - - - - -')

start_time = time.time()
filtered_img = cv.filter2D(img, ddepth=-1, kernel=kernel2d) # ddepth = -1이면 입력 이미지(img)와 동일한 깊이를 사용 -> 주의할 점 채널의 깊이가 아니라 이미지의 픽셀의 비트 깊이를 의미, 즉, 픽셀이 가질 수 있는 비트의 범위(비트 표현의 깊이)라고 볼 수 있음
end_time = time.time()
print('2D filtered 이미지 비트뎁스:',filtered_img.itemsize * 8,'비트')
print('2D filtered shape:',filtered_img.shape)
print(f"2D Gaussian Filtering 수행 시간: {end_time - start_time:.6f} 초")
cv.imwrite('./separable_filtered/2Dfiltered.jpg', filtered_img)
print('- - - - - - - - - - - - - - - - - - - - - - -')


# 4) 입력 영상을 kernel1d를 적용한 후 그 결과에 다시 kernel1d.transpose()를 적용하여 separable filtering을 수행
start_time = time.time()
kernel1d_filtered_img = cv.filter2D(img, ddepth=-1, kernel=kernel1d) # 1차 필터링 - 세로방향 kernel1d 적용
separable_filtered_img = cv.filter2D(kernel1d_filtered_img, ddepth=-1, kernel=kernel1d.transpose()) # 2차 필터링 - 가로방향 kernel1d.T 적용
end_time = time.time()
print('separable filtered 이미지 비트뎁스:',separable_filtered_img.itemsize * 8,'비트')
print('separable filtered 이미지 shape:',separable_filtered_img.shape)
print(f"Separable Filtering 수행 시간: {end_time - start_time:.6f} 초")
cv.imwrite('./separable_filtered/separable_filtered.jpg', separable_filtered_img)
print('- - - - - - - - - - - - - - - - - - - - - - -')

# 5) 필터링 결과 비교
# 두 배열의 모든 값이 정확히 동일한지 검사(부동 소수점 오차를 허용하지 않음.)
if np.array_equal(filtered_img, separable_filtered_img):
    print("두 필터링 결과가 정확히 동일합니다.")
else:
    print("두 필터링 결과가 다릅니다.")

# 두 배열의 값이 부동소수점 오차를 고려해 거의 동일한지 검사(부동 소수점 연산의 미세한 차이를 허용할 수 있음.)
if np.allclose(filtered_img, separable_filtered_img, atol=1e-6):
    print("두 필터링 결과가 거의 동일합니다.")
else:
    print("두 필터링 결과가 다릅니다.")

# sub_image = filtered_img - separable_filtered_img
sub_image = cv.absdiff(filtered_img, separable_filtered_img)
print('sub_image 이미지 비트뎁스:',sub_image.itemsize * 8,'비트')
print(sub_image.shape)
cv.imshow('Difference Image',sub_image)
cv.waitKey(0)
# cv.imshow('filtered_img Image',filtered_img)
# cv.waitKey(0)
# cv.imshow('separable_filtered_img Image',separable_filtered_img)
# cv.waitKey(0)
cv.destroyAllWindows()
