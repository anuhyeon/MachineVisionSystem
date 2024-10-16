import cv2 as cv
import numpy as np

def addsaltandpeppernoise(image, ratio):
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(image.size * ratio)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    out[coords[0],coords[1],coords[2]] = 255
    # Pepper mode
    num_pepper = np.ceil(image.size * ratio)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    out[coords[0],coords[1],coords[2]] = 0
    return out


# 원본 이미지 읽기 (그레이스케일)
img = cv.imread('./resized_images/resized_102811.jpg')

# 세 가지 비율로 Salt and Pepper Noise 추가
ratios = [0.02, 0.1, 0.25]

for ratio in ratios:
    noisy_img = addsaltandpeppernoise(img, ratio)
    cv.imwrite(f'./saltandpeppernoise/salt_pepper_noise_ratio_{ratio}.jpg', noisy_img)

print("Salt and Pepper Noise가 추가된 세 장의 이미지가 생성되었습니다.")
