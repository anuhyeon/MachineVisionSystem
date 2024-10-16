import cv2 as cv 
import numpy as np
import os

img = cv.imread('./resized_images/resized_102811.jpg')
blur = cv.blur(img,ksize=(7,7)) # 3,3 5,5 7,7
detail = np.int32(img) - np.int32(blur) # 원래의 이미지에서 부드러움을 뺌 = detail만 남음
cv.imwrite('./detail/detail_7x7.jpg', detail)
alpha = [2,5,10]

output_dir = './sharpened/'
for a in alpha:
    sharpened_img = np.int32(img) + a*detail
    # sharpened이미지 파일 저장 경로 설정
    output_path = os.path.join(output_dir, f'sharpened7x7_{a}.jpg')
    # 이미지 저장
    cv.imwrite(output_path, sharpened_img)
    print(f"{output_path}에 sharpened 이미지가 저장되었습니다.")
    
#     # 이미지 표시
#     cv.imshow(f'Filtered Image {k}x{k}', sharpened_img)
    
#     # 키 입력 대기 (0이면 무한 대기)
#     cv.waitKey(0)

# # 모든 창 닫기
# cv.destroyAllWindows()

