import torch
import torchvision.transforms as transforms
from PIL import Image
# 대상 디바이스 설정
import torch
import torch.nn as nn
from utils import *
from engine import engine
import utils
import train
from model_builder import *
import model_builder
from utils import calculate_accuracy, update_confusion_matrix, calculate_precision_recall,conf_to_df,plot_confusion_heatmap, calculate_and_plot_f1_scores,plot_loss,plot_acc,image_transform



def predict_function(model,model_pth: str,
                        img_path,
                        device: torch.device = "cpu"):
    output_list = []
    image = Image.open(img_path).convert("RGB")
    image_tensor = image_transform(image).to(device)
    data = image_tensor.unsqueeze(0)
    model = model(6)
    model.load_state_dict(torch.load(model_pth))
    model.to(device)
    model.eval()
    with torch.inference_mode():
        model.eval()
        output = model(data)
    return torch.sigmoid(output)

# if __name__ == "__main__":
#   pass
# model_pth = '/content/best_GoogLeNet.pth'
# output = predict_function(model_pth=model_pth,img_path = '/content/drive/MyDrive/개전체_100개_병별로/백내장/crop_D0_0da935dc-60a5-11ec-8402-0a7404972c70.jpg')
# output
