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




acc_model = {}
loss_model = {}

def main(model,model_name='model', num_classes=6, epochs=15):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # CUDA GPU가 사용 가능한 경우
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # M1/M2 칩이 있는 경우 MPS 사용
    else:
        device = torch.device("cpu")  # 그 외의 경우 CPU 사용

    print(f"Using device: {device}")
    # num_classes = 6  # 클래스의 개수 (정상과 6가지 질병)
    model = model(num_classes).to(device)

    # 손실 함수와 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(model_name=model_name,patience=5, delta=0.001, mode='min', verbose=True)

    # 엔진을 사용하여 훈련 시작 (engine.py)
    results = engine(model, criterion, optimizer, device, early_stopping, epochs=epochs, num_classes=num_classes)

    # acc_model, loss_model에 추가
    if model_name not in acc_model:
        acc_model[model_name] = []
        loss_model[model_name] = []

    # 리스트를 extend()를 사용하여 추가
    acc_model[model_name].extend(results['val_accuracies'])
    loss_model[model_name].extend(results['val_losses'])

    return results
