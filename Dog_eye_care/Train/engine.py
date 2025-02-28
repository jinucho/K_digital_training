# 학습 동작 함수
import numpy as np
from train import train, validate
from utils import *
from data_loader import data_loader
train_loader, valid_loader = data_loader()


def engine(model, criterion, optimizer, device, early_stopping,num_classes, epochs=15):
    model = model.to(device)
    criterion = criterion.to(device)
    best_val_score = np.Inf if early_stopping.mode == 'min' else 0

    results = {
    'train_losses' : [],
    'train_accuracies' : [],
    'val_losses' : [],
    'val_accuracies':[],
    'best_train_loss': None,
    'best_train_accuracy': None,
    'best_conf_matrix_train': None,
    'best_val_loss': None,
    'best_val_accuracy': None,
    'best_conf_matrix_val': None,
    'val_all_targets':None,
    'val_all_outputs':None}

    for epoch in range(epochs):
        train_loss, train_accuracy, conf_matrix_train = train(model, train_loader, optimizer, criterion, device,num_classes)
        val_loss, val_accuracy, conf_matrix_val, all_targets, all_outputs = validate(model, valid_loader, criterion, device,num_classes)
        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_accuracy)
        results['val_losses'].append(val_loss)
        results['val_accuracies'].append(val_accuracy)

        # 모든 에포크에서 모델 상태 저장
        early_stopping.save_checkpoint(val_loss, model, epoch)

        # EarlyStopping 호출 및 최고 성능 모델 저장
        early_stopping(val_loss)
        if early_stopping.mode == 'min' and val_loss < best_val_score:
            best_val_score = val_loss
            results['best_train_loss'] = train_loss
            results['best_train_accuracy'] = train_accuracy
            results['best_conf_matrix_train'] = conf_matrix_train
            results['best_val_loss'] = val_loss
            results['best_val_accuracy'] = val_accuracy
            results['best_conf_matrix_val'] = conf_matrix_val
            results['val_all_targets'] = all_targets
            results['val_all_outputs'] = all_outputs
            early_stopping.save_checkpoint(val_loss, model, epoch, is_best=True)
        elif early_stopping.mode == 'max' and val_loss > best_val_score:
            best_val_score = val_loss
            early_stopping.save_checkpoint(val_loss, model, epoch, is_best=True)

        if early_stopping.early_stop:
            print("조기 종료됨")
            break

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    return results

# if __name__ == "__main__":
#     engine()
