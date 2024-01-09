import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action='ignore')
from pycaret.classification import *
import pickle
import streamlit as st
import requests

about_info = """
NHIS2018 데이터 기반 당뇨 예측 모델
"""
st.sidebar.title("About")
st.sidebar.info(about_info)

logo_url = "https://www.biotimes.co.kr/news/photo/202204/7608_8719_340.jpg"
st.sidebar.image(logo_url)

# ref_df = pd.read_csv('설문지_CV.csv',index_col = 'Unnamed: 0')
url = 'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/ref_DataFrame.csv'
ref_df = pd.read_csv(url,index_col = 'Unnamed: 0')
pkl_list = ['ensemble_model.pickle', 'Scaler_AGE.pickle', 'Scaler_BMI.pickle', 'Scaler_HEIGHT(cm).pickle', 'Scaler_WEIGHT(kg).pickle' ]
val_list = ['model', 'AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']
scalers = ['AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']

# Load 피클 데이터 
list_dic = {}
val_list = ['model', 'AGE', 'BMI', 'HEIGHT(cm)', 'WEIGHT(kg)']
urls = [
    'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/pkl/ensemble_model.pickle',
    'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/pkl/Scaler_AGE.pickle',
    'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/pkl/Scaler_BMI.pickle',
    'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/pkl/Scaler_HEIGHT(cm).pickle',
    'https://raw.githubusercontent.com/jinucho/Mean_30/main/Streamlit_Service/pkl/Scaler_WEIGHT(kg).pickle'
]
for val,url in zip(val_list,urls):
    response = requests.get(url)
    data = response.content
    list_dic[val] = pickle.loads(data)


def validate_zero_one(text):
    return text if text in ['0', '1'] else None

def answerconvert(x):
    if x == '예' or x =='여성':
        return 1
    else : 
        return 0


st.title('당신은 당뇨 위험이 있을까?')

feature1 = st.text_input('이름을 입력하세요:', ) #-> ID로 사용
feature2 = st.text_input('나이를 입력하세요:', )



feature18 = st.text_input('키를 입력하세요:', )
feature19 = st.text_input('몸무게를 입력하세요:', )




feature3 = st.radio('음주를 자주 하나요?:',('예','아니오'))
feature3 = answerconvert(feature3)


feature4 = st.radio('관절염이 있나요:',('예','아니오'))
feature4 = answerconvert(feature4)
    

feature6 = st.radio('콜레스테롤이 높나요?:',('예','아니오'))
feature6 = answerconvert(feature6)

feature7 = st.radio('폐기종이 있나요?:',('예','아니오'))
feature7 = answerconvert(feature7)


feature8 = st.radio('균형잡힌 식사를 하나요?:',('예','아니오'))
feature8 = answerconvert(feature8)

feature9 = st.radio('성별을 선택하세요:',('남성','여성'))
feature9 = answerconvert(feature9)

feature11 = st.radio('고혈압이 있나요?:',('예','아니오'))
feature11 = answerconvert(feature11)

feature12 = st.radio('이전에 혈압약을 먹었나요?:',('예','아니오'))
feature12 = answerconvert(feature12)

feature13 = st.radio('현재 혈압약 먹나요?:',('예','아니오'))
feature13 = answerconvert(feature13)

feature14 = st.radio('최근 복통/구토/설사등 위장 장애가 있었나요?:',('예','아니오'))
feature14 = answerconvert(feature14)
    
feature10 = st.radio('히스패닉인가요?:',('예','아니오'))
feature10 = answerconvert(feature10)

    
race = {"백인": 1,"흑인": 2,"인도인": 3,"중국인":6,"필리핀":7,"아시아인":12,"혼혈":17,"이외":16}
selected_race = st.selectbox('인종을 선택해주세요:', list(race.keys()))
feature15 = race[selected_race]
feature16 = feature15
# region = {"동북": 1,"중서부": 2,"남부": 3,"서부":4}
# selected_region = st.selectbox('지역을 선택해주세요:', list(region.keys()))
# feature17 = region[selected_region]
feature17 = 1

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

if st.button('설문 결과'):
    try:
        feature5 = str(int(int(feature19)/((int(feature18)/100)**2)))
        col_text = 'id AGE ALCSTAT ARTH1 BMI CHLEV EPHEV FSBALANC GENDER HISPAN_I HYPEV HYPMDEV2 HYPMED2 INTIL2W MRACBPI2 MRACRPI2 REGION HEIGHT(cm) WEIGHT(kg) DIBEV1'
        columns = col_text.split(' ')
        user_input_feature = [feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,feature17,feature18,feature19]
        data = dict(zip(columns,user_input_feature))
        input_df = pd.DataFrame(data,index=[0])
        st.write('설문결과')
        st.dataframe(input_df)
        if st.button('당신은 당뇨일까?!'):
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
            st.info(f"결과 : {feature1}님은 현재 {result} 입니다. 또한, 당뇨일 확률은 {percent}% 입니다.")
    except:
        st.write('누락된 값이 있습니다.')



    
