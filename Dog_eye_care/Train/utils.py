# util함수 및 Early_stopping
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import precision_recall_curve


def calculate_accuracy(outputs, targets, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    correct = (preds == targets).float()
    accuracy = correct.sum() / (targets.size(0) * targets.size(1))  # 전체 레이블 수로 나눔
    return accuracy



class EarlyStopping:
    def __init__(self,model_name='model', patience=10, delta=0.001, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.model_name = model_name

        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # CUDA GPU가 사용 가능한 경우
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # M1/M2 칩이 있는 경우 MPS 사용
        else:
            self.device = torch.device("cpu")  # 그 외의 경우 CPU 사용


    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':# 기준값이 낮아지는걸 개선이라고 한다. cost, loss
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')

        elif self.mode == 'max':# 기준값이 높아지는걸 개선이라고 한다. accuracy
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')


        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
    def save_checkpoint(self, score, model, epoch, is_best=False):
        '''성능이 향상될 때 모델을 저장합니다.
        is_best가 True일 경우, 모델을 'best_model.pth'로 저장합니다.'''
        if self.verbose:
            print(f'Saving {self.model_name} at epoch {epoch+1} with score: {score:.5f}')
        filename = f'epoch_{epoch+1}_{self.model_name}.pth' if not is_best else f'best_{self.model_name}.pth'
        torch.save(model.to('cpu').state_dict(), filename)
        model.to(self.device)


def update_confusion_matrix(conf_matrix, outputs, targets, threshold=0.5):
    preds = outputs > threshold  # 임계값을 기준으로 이진 예측값 생성
    for i in range(targets.shape[1]):  # 각 레이블에 대해
        # True Positives
        conf_matrix[i, i] += torch.sum((preds[:, i] == 1) & (targets[:, i] == 1)).item()
        # False Positives
        for j in range(targets.shape[1]):
            if i != j:
                conf_matrix[i, j] += torch.sum((preds[:, i] == 1) & (targets[:, j] == 1)).item()
    return conf_matrix



def calculate_precision_recall(conf_matrix):
    num_labels = conf_matrix.size(0)
    precisions = []
    recalls = []

    for i in range(num_labels):
        tp = conf_matrix[i, i].item()
        fp = (conf_matrix[:, i].sum() - tp).item()
        fn = (conf_matrix[i, :].sum() - tp).item()
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    df = pd.DataFrame({'Disease': ['결막염','궤양석각막질환','백내장','색소침착성각막염','안검종양','유루증'], 'Precision': precisions, 'Recall': recalls})
    return df

def conf_to_df(conf_matrix):
  disease = ['결막염', '궤양석각막질환', '백내장', '색소침착성각막염', '안검종양', '유루증']

  # 멀티인덱스 생성
  # multi_index = pd.MultiIndex.from_product([['실제'], disease], names=['', '질병'])
  multi_index = pd.MultiIndex.from_product([['실제'], disease])
  multi_columns = pd.MultiIndex.from_product([['예측'], disease])
  # multi_columns = pd.MultiIndex.from_product([['예측'], disease], names=['', '질병'])

  # 데이터프레임으로 변환
  cm_to_df = pd.DataFrame(conf_matrix.numpy(), index=multi_index, columns=multi_columns)
  return cm_to_df


def plot_confusion_heatmap(results, title='혼동행렬'):
    """
    히트맵 형식의 혼동 행렬을 그리는 함수.

    Args:
        confusion_matrix (np.ndarray): 혼동 행렬 데이터 (2D 배열)
        title (str): 히트맵의 제목

    Returns:
        None
    """
    confusion_matrix = results['best_conf_matrix_val']
    # 레이블 정의
    labels = ['결막염', '궤양석각막질환', '백내장', '색소침착성각막염', '안검종양', '유루증']

    # 히트맵 색상 설정
    cmap = sns.color_palette("Blues", as_cmap=True)

    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=".1f", cmap=cmap, linewidths=.5, cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.show()

def calculate_and_plot_f1_scores(results,title='평균 F1 Score'):
    labels = ['결막염', '궤양성각막질환', '백내장', '색소침착성각막염', '안검종양', '유루증']
    all_f1_scores = []

    plt.figure(figsize=(10, 10))
    max_f1_scores_info = ""  # 최대 F1 점수 정보를 저장할 문자열

    for i, label in enumerate(labels):
        precision, recall, thresholds = precision_recall_curve(results['val_all_targets'][:, i], results['val_all_outputs'][:, i])

        # 임계값이 0부터 1 사이인 부분만 선택
        valid_indices = np.where((thresholds >= 0) & (thresholds <= 1))
        valid_thresholds = thresholds[valid_indices]
        valid_precision = precision[valid_indices]
        valid_recall = recall[valid_indices]

        # F1 점수 계산
        f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall)
        f1_scores = np.nan_to_num(f1_scores)  # NaN 값을 제거

        # 레이블별 최대 F1 점수 찾기
        max_f1_index = np.argmax(f1_scores)
        max_f1_score = f1_scores[max_f1_index]
        max_threshold = valid_thresholds[max_f1_index]

        # 최대 F1 점수 정보 추가
        max_f1_scores_info += f'{label}: F1={max_f1_score:.2f}, Threshold={max_threshold:.2f}\n'

        # 모든 레이블에 대한 F1 점수를 리스트에 추가
        all_f1_scores.append(f1_scores)

    # 모든 레이블에 대한 평균 F1 점수 계산
    mean_f1_scores = np.mean(all_f1_scores, axis=0)

    # 임계값을 0부터 1까지 설정
    valid_thresholds = np.linspace(0, 1, len(mean_f1_scores))

    # 평균 F1 점수 그래프 출력
    plt.plot(valid_thresholds, mean_f1_scores, lw=2, label='평균 F1 Score', color='orange')

    # 최대 F1 점수 정보 표시
    plt.text(0.15, 0.1, max_f1_scores_info, fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend(loc="best")
    plt.show()


def plot_loss(train_loss, val_loss, model_name='model'):
  plt.plot(train_loss, '-',c='r')
  plt.plot(-np.array(val_loss), '--',c='b')
  plt.title(model_name)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['train','val'])
  plt.ylim(-0.6, 0.6)
  plt.show()

def plot_acc(train_acc, val_acc, model_name='model'):
  plt.plot(train_acc, '-',c='r')
  plt.plot(-np.array(val_acc), '--',c='b')
  plt.title(model_name)
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['train','val'])
  plt.ylim(0.75, 1.0)
  plt.show()

# Transform 조합 생성
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
