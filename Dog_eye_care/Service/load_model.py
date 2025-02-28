from PIL import Image
from io import BytesIO
import time
import os
import json
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import jsonify

import torchvision.models as models
import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')

from openai import OpenAI

my_key = 'my_code'
client = OpenAI(api_key=my_key)

def preprocessing_byte(byte_image):
    decoded = base64.b64decode(byte_image)
    IMAGE_SHAPE = (224, 224)
    user_input_image = Image.open(BytesIO(decoded)).resize(IMAGE_SHAPE)
    return user_input_image

from flask import jsonify

def query_openai(results):
    # 각 원소를 스칼라로 변환
    results_numpy = [
        {
            "result": (result[0] * 100).round().detach().cpu().numpy().tolist()
        }
        for result in results
    ]

    # results_numpy 리스트에서 확률 정보 추출
    symptoms = [
        f'결막염 : {result["result"][0]}%, 유루증 : {result["result"][1]}%, 궤양성각막질환 : {result["result"][2]}%, 백내장 : {result["result"][3]}%, 색소침착성각막염 : {result["result"][4]}%, 안검종양 : {result["result"][5]}%'
        for result in results_numpy
    ]

    # 결과를 OpenAI API에 전달
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "너는 아주 유능한 강아지 안구질병을 판단하는 어플이야."},
            {"role": "user",
             "content": f"""강아지 안구 사진으로 진단 받았을 때 각 안구 질병의 확률이 아래와 같아 : {symptoms[0]}, {symptoms[1]}, 각 질병에 대한 아주 짧은 설명과 어떻게 관리하면 좋을 지 안내해줘. 각 질병 퍼센트를 ()안에 두개를 반드시 표시해. 각 설명 마지막에 _을 써줘"""}
        ]
    )

    # 파일 경로 설정
    # file_path = "output.json"
    #
    # # JSON 데이터를 파일에 쓰기
    # with open(file_path, "w") as json_file:
    #     json.dump(response.choices[0].message.content, json_file)

    return jsonify(response.choices[0].message.content)


# def query_openai(result):
#     result = (result[0]*100).round()
#     # 작업 완료 후 결과 표시
#     symptoms = f'결막염 : {result[0]}%, 유루증 : {result[1]}%, 궤양성각막질환 : {result[2]}%, 백내장 : {result[3]}%, 색소침착성각막염 : {result[4]}%, 안검종양 : {result[5]}%'
#     response = client.chat.completions.create(
#       model="gpt-4-0125-preview",
#       messages=[
#           {"role": "system", "content": "너는 아주 유능한 강아지 안구질병을 판단하는 어플이야."},
#           {"role": "user", "content": f"""강아지 안구 사진으로 진단 받았을 때 각 안구 질병의 확률이 아래와 같아 : {symptoms}, 각 질병에 대한 아주 짧은 설명과 어떻게 관리하면 좋을 지 안내해줘"""}
#       ]
#     )
#
#     return response.choices[0].message.content

# def query_openai(results):
#     # 각 원소를 스칼라로 변환
#     results_numpy = [(result[0]*100).round().detach().cpu().numpy().tolist() for result in results]
#     return jsonify(result=results_numpy)



class GoogLeNetModel(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNetModel, self).__init__()
        # 사전 훈련된 GoogLeNet 모델 로드
        self.googlenet = models.googlenet(pretrained=True)
        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)



def image_transform(x):
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(30),
      transforms.ColorJitter(brightness=0.5),
      transforms.RandomVerticalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return transform(x)

def predict_function(image, model_pth: str = 'st/model_inference/best_GoogLeNet.pth', device: torch.device = "cpu"):
    image = image.convert('RGB')
    image_tensor = image_transform(image).to(device)
    data = image_tensor.unsqueeze(0)
    model = GoogLeNetModel(6)
    model.load_state_dict(torch.load(model_pth))
    model.to(device)
    model.eval()
    with torch.inference_mode():
        model.eval()
        output = model(data)
    return torch.sigmoid(output)


def save_image_to_pic_folder(image, file_name):
    # 현재 작업 디렉토리에 'pic' 폴더 생성
    folder_path = "yolo_pic"
    create_folder_if_not_exists(folder_path)

    # 'pic' 폴더에 파일 저장
    file_path = os.path.join(folder_path, f"{file_name}.png")

    # 이미지를 바이트로 변환
    image_bytes = image_to_bytes(image)

    # 바이트를 파일에 쓰기
    with open(file_path, "wb") as file:
        file.write(image_bytes)


def image_to_bytes(image):
    with BytesIO() as byte_io:
        image.save(byte_io, format="PNG")
        return byte_io.getvalue()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)




def detect_eye_diseases(image, model_pth: str = 'st/model_inference/best_ViT.pth'):
    result = predict_function(model_pth,image)
    print(result)
