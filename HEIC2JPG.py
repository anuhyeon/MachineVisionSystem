import os
import pyheif
from PIL import Image

def heic_to_jpg(input_dir, output_dir):
    # 입력 및 출력 디렉토리가 존재하는지 확인하고 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리에서 모든 HEIC 파일 찾기
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".heic"):
            heic_file_path = os.path.join(input_dir, filename)
            jpg_filename = filename.rsplit(".", 1)[0] + ".jpg"
            jpg_file_path = os.path.join(output_dir, jpg_filename)
            
            # HEIC 파일을 읽어서 JPG로 변환
            heif_file = pyheif.read(heic_file_path)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            
            # JPG 파일로 저장
            image.save(jpg_file_path, "JPEG")
            print(f"{filename} -> {jpg_filename} 변환 완료!")
            
            # HEIC 파일 삭제
            os.remove(heic_file_path)
            print(f"{filename} 삭제 완료!")

# 입력 디렉토리와 출력 디렉토리 설정 상대 경로를 쓰니깐 에러가 발생함.
# input_directory = "./HEIC"  # HEIC 파일이 있는 디렉토리
# output_directory = "./JPG"  # 변환된 JPG 파일을 저장할 디렉토리 

input_directory = "/Users/an-uhyeon/MachineVisionSystem/HEIC"  # 절대 경로 사용
output_directory = "/Users/an-uhyeon/MachineVisionSystem/images"


# 변환 함수 실행
heic_to_jpg(input_directory, output_directory)
