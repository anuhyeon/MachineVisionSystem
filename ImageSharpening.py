import cv2 as cv 
import numpy as np
import os

img = cv.imread('./resized_images/resized_102811.jpg')
blur = cv.blur(img,ksize=(7,7)) # (3,3) (5,5) (7,7) -> 각각 경우에 대해 ksize를 바꿔가며 코드를 총 3번 실행시킬 것.
detail = np.int32(img) - np.int32(blur) # 원래의 이미지에서 부드러움을 뺌 = detail만 남음
cv.imwrite('./detail/detail_7x7.jpg', detail)
alpha = [2,5,10]

output_dir = './sharpened/'
for a in alpha:
    sharpened_img = np.int32(img) + a*detail # 원본 이미지에서 detail을 더 추가하여 영상을 좀 더 샤프하게 만든다.
    output_path = os.path.join(output_dir, f'sharpened7x7_{a}.jpg')
    cv.imwrite(output_path, sharpened_img)
    print(f"{output_path}에 sharpened 이미지 저장 완료.")


