import cv2

# 이미지 파일 읽기
image_path = "/Users/an-uhyeon/MachineVisionSystem/resized_images/resized_im_dst_poster.jpg"  # 이미지 파일 경로
image = cv2.imread(image_path)

if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# 좌표를 출력하는 함수
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭
        print(f"좌표: x={x}, y={y}")

# 창 이름 설정 및 마우스 이벤트 연결
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_coordinates)

# 이미지 창 열기
while True:
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

cv2.destroyAllWindows()
