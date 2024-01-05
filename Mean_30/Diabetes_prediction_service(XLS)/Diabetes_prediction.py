from pycaret.classification import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import openpyxl

pkl_list = ['ensemble_model.pickle', 'Scaler_AGE.pickle', 'Scaler_BMI.pickle', 'Scaler_HEIGHT(cm).pickle', 'Scaler_WEIGHT(kg).pickle' ]
val_list = ['model', 'AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']
scalers = ['AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']

# Load 피클 데이터 
list_dic = {}
for index in range(len(pkl_list)):
    with open(f"pkl\\{pkl_list[index]}", "rb") as fr:
        list_dic[val_list[index]] = pickle.load(fr)

# Load 설문지
docs = pd.read_excel('설문지.xlsx', sheet_name='df', engine='openpyxl')

# Scaler 생성(from pickle) 할당 및 컬럼 Scaling
for s in scalers:
    scale = list_dic[s]
    docs[s] = scale.transform(docs[[s]])

# 모델 생성(from pickle)
model = list_dic['model']

# 당뇨병 예측
y_pred = model.predict(docs)
y_prob = model.predict_proba(docs)[:,y_pred]

y_pred = y_pred[0]
y_prob = round(y_prob[0][0]*100,2)

if y_pred == 1:
    result ="당뇨"
    if y_prob >= 0.5:
        a= f"당뇨일 가능성이 높습니다. ({y_prob} %)"
    else:
        a= f"당뇨 가능성이 있습니다. 추가검진이 필요합니다. ({y_prob} %)"
else:
    result ="정상"
    if y_prob >= 0.5:
        a= f"정상이신것 같습니다. ({y_prob} %)"
    else:
        a= f"정상이나 관리가 필요합니다. ({y_prob} %)"

print(a)
with open(f"Diabetes_Result_is_■{result}■.txt", "w", encoding="UTF-8") as f:
    f.write(a)
    
