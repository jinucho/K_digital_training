import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from pycaret.classification import *
import pickle

def diabetes_MachineModeling(path: str=None, Data_preprocessing: str=None,):

    if path == None:
        df = pd.read_csv('Data_preprocessing.csv')
    else:
        df = pd.read_csv(f'{path}\\{Data_preprocessing}.csv')

    # 2. 학습 / 테스트 데이터 분리
    X_tr, X_test, y_tr, y_test = train_test_split(df.drop(['id','DIBEV1'],axis=1),df['DIBEV1'],shuffle=True,test_size=0.05,stratify=df['DIBEV1'],random_state=123)
    X_test_org = X_test.copy()

    # 3. 불균형 데이터 처리(언더샘플링)
    sampler = RandomUnderSampler(random_state=123)
    X_under , y_under = sampler.fit_resample(X_tr,y_tr)

    # 4. 수치형 변수 데이터 스케일링 대상 컬럼 선정
    num_col = ['AGE','BMI','HEIGHT(cm)','WEIGHT(kg)']

    # 전체 데이터셋의 위 컬럼들에 대해 스케일링 fit 후 각 데이터셋 및 검증용 데이터셋에 transform 적용
    for col in num_col:
        ss = StandardScaler()
        X_under[col] = ss.fit_transform(X_under[[col]])
        X_test[col] = ss.transform(X_test[[col]])
        
        with open(f"pkl\\Scaler_{col}.pickle", "wb") as fw:
            pickle.dump(ss, fw)
        
        
    # 5. 범주형 변수 onehotencoding
    # 데이터프레임 전체 컬럼에서 수치형 컬럼 제외
    cols = np.setdiff1d(X_under.columns,num_col)

    # 위 컬럼에서 고유값 개수가 3개 이상인 컬럼만 추출
    # 0,1만 가지는 binary 컬럼은 굳이 ohe를 하지 않을 것
    nom_col = [col for col in cols if X_under[col].nunique() >= 3 ]

    # 명목형 컬럼들에 대한 dummy 데이터 생성(원핫인코딩)
    train_dummies = [] # 학습용 데이터셋의 명목형 컬럼들의 더미데이터셋 저장용 리스트
    test_dummies = [] # 검증용 데이터셋의 명목형 컬럼들의 더미데이터셋 저장용 리스트
    for col in nom_col:
        train_dummies.append(pd.get_dummies(X_under[col],prefix=col,dtype='int')) # 학습데이터의 각 컬럼들의 더미데이터셋을 리스트에 저장
        test_dummies.append(pd.get_dummies(X_test[col],prefix=col,dtype='int')) # 검증데이터의 각 컬럼들의 더미데이터셋을 리스트에 저장

    train_dummies = pd.concat(train_dummies,axis=1) # 학습 데이터의 더미데이터셋 리스트를 하나로 합침
    test_dummies = pd.concat(test_dummies,axis=1) # 검증 데이터의 더미데이터셋 리스트를 하나로 합침

    # 만약 고유값 개수 차이로 인해 학습셋과 테스트셋의 더미 데이터셋 컬럼 차이가 있다면 컬럼수를 통일
    # 학습 및 예측 오류 방지

    if train_dummies.columns.nunique() > test_dummies.columns.nunique():
        missing_cols = set(train_dummies.columns) - set(test_dummies.columns)
        for col in missing_cols:
            test_dummies[col] = 0
    elif train_dummies.columns.nunique() < test_dummies.columns.nunique():
        missing_cols = set(test_dummies.columns) - set(train_dummies.columns)
        for col in missing_cols:
            train_dummies[col] = 0
    else:
        pass
    
    test_dummies = test_dummies[train_dummies.columns]

    # 원본의 학습,테스트셋에 더미데이터셋 합친 후 기존 명목형 컬럼 제거
    X_under = pd.concat([X_under,train_dummies],axis=1).drop(nom_col,axis=1)
    X_test = pd.concat([X_test,test_dummies],axis=1).drop(nom_col,axis=1)

    # # Test Data pickle 저장
    # with open("X_test.pickle", "wb") as fwx:
    #     pickle.dump(X_test, fwx)

    # with open("y_test.pickle", "wb") as fwy:
    #     pickle.dump(y_test, fwy)
        
    
    # 6. 전처리 완료 후 학습 / 검증 데이터 분리
    X_tr, X_val, y_tr, y_val = train_test_split(X_under,y_under,test_size=0.2,random_state=123,shuffle=True,stratify=y_under)
    
    # 7. pycaret 최적화 모델링 : ensemble_model
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    clf = setup(data=X_tr,target=y_tr,preprocess=False,verbose=False,n_jobs=-1,session_id=123) # pycaret AutoML사용을 위한 초기화
    best_5 = compare_models(fold=10, sort='auc',verbose=False,n_select=5) # pycaret에서 sort에 설정한 평가지표 기준으로 데이터셋에 최적화된 모델 선정(n_select = 1이 기본값이며, 이 값에 따라 선정되는 모델의 개수 변경 됨)
    tuned_models = []
    for model in best_5:
        tuned_model = tune_model(model,optimize='auc',verbose=False,search_library='optuna',fold=skf)#,search_algorithm='optuna')
        tuned_models.append(tuned_model)

    ensemble_model = blend_models(estimator_list=tuned_models, method='auto',optimize='auc',verbose=False) # best_5 모델들에 대한 앙상블
    save_model(ensemble_model, 'ensemble_model')
    
    # 8. 최종 X_test 모델성능 확인
    pred = predict_model(ensemble_model,data=X_test,verbose=True)

    # 각 test데이터 별 결정 된 label마다의 확률이 저장 된 prediction_score를 기반으로 label 0,1 중 1에 대한 확률값을 가진 컬럼 정의 
    pred['predict_proba'] = pred['prediction_score']
    pred.loc[pred['prediction_label']==0,'predict_proba'] = 1-pred.loc[pred['prediction_label']==0,'predict_proba']

    ## 9. 예측 확률값을 임계값 기준으로 최종 예측 값 정리
    j = 0.5
    result = [0 if i<=j else 1 for i in pred['predict_proba']]

    # 10. 예측 결과
    result_str = ['정상' if i==0 else '당뇨' for i in result]
    y_test_str = ['정상' if i==0 else '당뇨' for i in y_test]
    model_report  = classification_report(y_test_str,result_str)
    with open("model_report.txt", "w") as f:
        for letters in model_report:
            f.write(f"{letters}")
    


