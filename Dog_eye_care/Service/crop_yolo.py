import torch
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

def estimate_blur(image, model_path = 'C:\\Users\\user\\PycharmProjects\\pythonProject\\st\\model_inference\\last.pt', threshold=100):
    
    # 이미지를 그레이스케일로 변환합니다
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 라플라시안 변환을 적용합니다
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 라플라시안 변환의 분산을 계산합니다
    variance = laplacian.var()
    
    # threshold값 기준으로 흐림/선명 판별 100보다 크면 선명
    if variance > threshold:
        print(f"이미지가 선명합니다. (variance = {variance}).")
        return crop_image(image, model_path)
    else:
        print(f"이미지가 흐립니다. (variance = {variance}).")
        return False
def crop_image(image, model_path):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 이미지에서 검출 수행
    results = model(image)

    # 잘린 이미지를 저장할 리스트
    cropped_images = []

    # 검출 결과 처리
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        # 바운딩 박스 좌표를 기반으로 이미지 자르기
        for box in boxes:
            xmin, ymin, xmax, ymax = box.astype(int)
            # 이미지를 BGR에서 RGB로 변환하여 PIL 이미지로 만듭니다.
            cropped_image = Image.fromarray(cv2.cvtColor(image[ymin:ymax, xmin:xmax], cv2.COLOR_BGRA2BGR))
            cropped_images.append(cropped_image)

    return cropped_images

# 실행코드
# image_path = 'testtest.jpg'
# image = cv2.imread(image_path)
# a = estimate_blur(image, 'last.pt')