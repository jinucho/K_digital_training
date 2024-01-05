import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')
from pycaret.classification import *
import pickle
import streamlit as st


about_info = """
NHIS2018 데이터 기반 당뇨 예측 모델
"""
st.sidebar.title("About")
st.sidebar.info(about_info)

logo_url = "https://www.biotimes.co.kr/news/photo/202204/7608_8719_340.jpg"
st.sidebar.image(logo_url)



ref_df = pd.read_csv('설문지_CV.csv',index_col = 'Unnamed: 0')
pkl_list = ['ensemble_model.pickle', 'Scaler_AGE.pickle', 'Scaler_BMI.pickle', 'Scaler_HEIGHT(cm).pickle', 'Scaler_WEIGHT(kg).pickle' ]
val_list = ['model', 'AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']
scalers = ['AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']

# Load 피클 데이터 
list_dic = {}
for index in range(len(pkl_list)):
    with open(f"pkl/{pkl_list[index]}", "rb") as fr:
        list_dic[val_list[index]] = pickle.load(fr)    
def validate_zero_one(text):
    return text if text in ['0', '1'] else None


st.title('당신은 당뇨 위험이 있을까?')

feature1 = st.text_input('이름을 입력하세요:', ) #-> ID로 사용
feature2 = st.text_input('나이를 입력하세요:', )
# feature3 = st.number_input('음주를 자주 하나요?(아니오:0,예:1):', min_value=0, max_value=1,step=1)

feature3 = st.text_input('음주를 자주 하나요?:')
if feature3 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature3 = int(feature3)
feature4 = st.text_input('관절염이 있나요?:')
if feature4 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature4 = int(feature4)
    
feature5 = st.text_input('BMI수치를 입력하세요(모르시면0):', )
feature18 = st.text_input('키를 입력하세요:', )
feature19 = st.text_input('몸무게를 입력하세요:', )

if feature5 == '0':
    try:
        feature5 = str(float(feature19)/((float(feature18)/100)**2))
    except:
        pass

feature6 = st.text_input('콜레스테롤이 높나요?:')
if feature6 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature6 = int(feature6)
    
feature7 = st.text_input('폐기종이 있나요?:')
if feature7 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature7 = int(feature7)
feature8 = st.text_input('균형잡힌 식사를 하나요?:')
if feature8 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature8 = int(feature8)
feature9 = st.text_input('성별을 입력하세요:')
if feature9 not in ['0', '1']:
    st.info('남성:0,여성:1')
else:
    feature9 = int(feature9)
    
feature11 = st.text_input('고혈압이 있나요?')
if feature11 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature11 = int(feature11)
    
feature12 = st.text_input('이전에 혈압약을 먹었나요?')
if feature12 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature12 = int(feature12)
    
feature13 = st.text_input('현재 혈압약 먹나요?')
if feature13 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature13 = int(feature13)
    
feature14 = st.text_input('최근 복통/구토/설사등 위장 장애가 있었나요?')
if feature14 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature14 = int(feature14)
    
feature10 = st.text_input('히스패닉인가요?')
if feature10 not in ['0', '1']:
    st.info('아니오:0,예:1')
else:
    feature10 = int(feature10)
    
race = {"백인": 1,"흑인": 2,"인도인": 3,"중국인":6,"필리핀":7,"아시아인":12,"혼혈":17,"이외":16}
selected_race = st.selectbox('인종을 선택해주세요:', list(race.keys()))
feature15 = race[selected_race]
feature16 = feature15
region = {"동북": 1,"중서부": 2,"남부": 3,"서부":4}
selected_region = st.selectbox('지역을 선택해주세요:', list(region.keys()))
feature17 = region[selected_region]






col_text = 'id AGE ALCSTAT ARTH1 BMI CHLEV EPHEV FSBALANC GENDER HISPAN_I HYPEV HYPMDEV2 HYPMED2 INTIL2W MRACBPI2 MRACRPI2 REGION HEIGHT(cm) WEIGHT(kg) DIBEV1'
columns = col_text.split(' ')
user_input_feature = [feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,feature17,feature18,feature19]
data = dict(zip(columns,user_input_feature))
input_df = pd.DataFrame(data,index=[0])
train_col = ['AGE', 'ALCSTAT', 'ARTH1', 'BMI', 'CHLEV', 'EPHEV', 'FSBALANC',
       'GENDER', 'HYPEV', 'HYPMDEV2', 'HYPMED2', 'INTIL2W', 'HEIGHT(cm)',
       'WEIGHT(kg)', 'HISPAN_I_0.0', 'HISPAN_I_1.0', 'HISPAN_I_2.0',
       'HISPAN_I_3.0', 'HISPAN_I_4.0', 'HISPAN_I_5.0', 'HISPAN_I_6.0',
       'HISPAN_I_7.0', 'HISPAN_I_8.0', 'HISPAN_I_12.0', 'MRACBPI2_1.0',
       'MRACBPI2_2.0', 'MRACBPI2_3.0', 'MRACBPI2_6.0', 'MRACBPI2_7.0',
       'MRACBPI2_12.0', 'MRACBPI2_16.0', 'MRACBPI2_17.0', 'MRACRPI2_1.0',
       'MRACRPI2_2.0', 'MRACRPI2_3.0', 'MRACRPI2_9.0', 'MRACRPI2_10.0',
       'MRACRPI2_11.0', 'MRACRPI2_15.0', 'MRACRPI2_16.0', 'MRACRPI2_17.0',
       'REGION_1.0', 'REGION_2.0', 'REGION_3.0', 'REGION_4.0']

if st.button('당신은 당뇨일까?!'):
    try:
        num_col = ['AGE','BMI','HEIGHT(cm)','WEIGHT(kg)']
        cat_col = ['HISPAN_I', 'MRACBPI2', 'MRACRPI2', 'REGION']
        nom_col = np.setdiff1d(input_df.drop('id',axis=1).columns,num_col,cat_col)
        input_df.loc[:,num_col] = input_df.loc[:,num_col].astype('float')
        input_df.loc[:,nom_col] = input_df.loc[:,nom_col].astype('int')

        cust_id = input_df.pop('id')

        # Scaler 생성(from pickle) 할당 및 컬럼 Scaling
        for s in scalers:
            scale = list_dic[s]
            input_df[s] = scale.transform(input_df[[s]])
        # 명목형 컬럼들에 대한 dummy 데이터 생성(원핫인코딩)
        test_dummies = [] # 검증용 데이터셋의 명목형 컬럼들의 더미데이터셋 저장용 리스트
        for col in cat_col:
            test_dummies.append(pd.get_dummies(input_df[col],prefix=col,dtype='int')) # 검증데이터의 각 컬럼들의 더미데이터셋을 리스트에 저장

        test_dummies = pd.concat(test_dummies,axis=1)    
        input_df = pd.concat([input_df,test_dummies],axis=1).drop(cat_col,axis=1)


        for col in train_col:
            if col not in input_df.columns:
                input_df[col] = 0

        # for col in input_df.columns:
        #     if col not in train_col:
        #         input_df.drop(col,axis=1)

        input_df = input_df[train_col]
        # st.write(f'{input_df.columns}')#,{y_prob}')
        # 모델 생성(from pickle)
        model = list_dic['model']

    #     # 당뇨병 예측
        y_pred = model.predict(input_df)
        y_prob = model.predict_proba(input_df)[:,1]
        result = '1'
        if y_pred == 0:
            result = '정상'
        else : 
            result = '당뇨'

        percent = str(round(y_prob[0],2)*100)
#         # y_prob = round(y_prob[0][0]*100,2)   
#         st.write(f'''{feature1}님은 현재 {result} 입니다.
        
        
# 또한, 당뇨일 확률은 {percent}% 입니다.''')
        st.sidebar.title("결과 : ")
        st.sidebar.info(f'''{feature1}님은 현재 {result} 입니다.
        
        
또한, 당뇨일 확률은 {percent}% 입니다.''')
    except:
         st.write('누락된 값이 있습니다, 입력 값을 확인하세요.')
    
