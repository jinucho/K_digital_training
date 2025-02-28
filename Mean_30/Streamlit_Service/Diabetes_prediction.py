import pandas as pd
import pickle

pkl_list = ['ensemble_model.pickle', 'Scaler_AGE.pickle', 'Scaler_BMI.pickle', 'Scaler_HEIGHT(cm).pickle', 'Scaler_WEIGHT(kg).pickle' ]
val_list = ['model', 'AGE', 'BMI', 'pickle', 'pickle']

list_dic = {}
for index in range(len(pkl_list)):
    with open(f"pkl\\{pkl_list[index]}", "rb") as fr:
        list_dic[val_list[index]] = pickle.load(fr)


df.read_csv("설문지.csv")