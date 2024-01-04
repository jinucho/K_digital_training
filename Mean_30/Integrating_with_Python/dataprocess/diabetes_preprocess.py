def diabetes_preprocess(path: str=None, samadult: str=None, samchild: str=None, familyxx: str=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    if path == None:
        df_SA = pd.read_csv("data\\samadult.csv")
        df_SC = pd.read_csv("data\\samchild.csv")
        df_FM = pd.read_csv("data\\familyxx.csv")
    else: 
        df_SA = pd.read_csv(f"{path}\\{samadult}.csv")
        df_SC = pd.read_csv(f"{path}\\{samchild}.csv")
        df_FM = pd.read_csv(f"{path}\\{familyxx}.csv")

    # 1.1 Selectd Column List
    with open("data\\Adult_col.txt", "r", encoding="UTF-8") as f:
        Adult_col = f.read()
        Adult_col = Adult_col.split('\n')
        
    with open("data\\Adult_newcol.txt", "r", encoding="UTF-8") as f:
        Adult_newcol = f.read()
        Adult_newcol = Adult_newcol.split('\n')

    with open("data\\SC_col.txt", "r", encoding="UTF-8") as f:
        SC_col = f.read()
        SC_col = SC_col.split('\n')

    with open("data\\SC_newcol.txt", "r", encoding="UTF-8") as f:
        SC_newcol = f.read()
        SC_newcol = SC_newcol.split('\n')
        
    with open("data\\fam_col.txt", "r",  encoding="UTF-8") as f:
        fam_col = f.read()
        fam_col = fam_col.split('\n')

    Adult_newcol = dict(zip(Adult_col,Adult_newcol))
    SC_newcol = dict(zip(SC_col,SC_newcol))

    df_SA = df_SA[Adult_col].rename(columns = Adult_newcol)
    df_SC = df_SC[SC_col].rename(columns = SC_newcol)
    df_FM = df_FM[fam_col]

    df_SA['id'] = '1000' + df_SA['HHX'].astype('str') + df_SA['FMX'].astype('str')
    df_SA.drop(['HHX','FMX'],inplace=True,axis=1)
    df_SC['id'] = '1000' + df_SC['HHX'].astype('str') + df_SC['FMX'].astype('str')
    df_SC.drop(['HHX','FMX'],inplace=True,axis=1)
    df_FM['id'] = '1000' + df_FM['HHX'].astype('str') + df_FM['FMX'].astype('str')
    df_FM.drop(['HHX','FMX'],inplace=True,axis=1)    
    
    # family 데이터프레임에 adult 데이터 프레임 merge
    temp = pd.merge(df_FM,df_SA,how='left',on='id')
    # combine_first() 를 이용해 병합
    total = temp.combine_first(df_SC)
    ## 2. 전처리(필터링, 결측치 처리)
    # 필터링 : child 12세 이상 필터링
    total = total[total['AGE'] >= 12]

    # 필터링 : 고혈압 yes(1)만 남기고 나머지는 전부 no(2 or 0)으로 처리
    total['HYPMDEV2'] = total['HYPMDEV2'].apply(lambda x: x if x == 1.0 else 2)

    # 필터링 : 아스피린 컬럼 제거 (상관성 없음)
    total = total.drop(['INSLN1','AFLHC24_','ASPMDMED','ASPMEDAD','ASPMEDEV','ASPONOWN'],axis=1)

    # 몸무게, 키 필터링
    # 1) 필터링 : 원본 키 인치 96~99는 지움
    total = total[(total['HEIGHT'] < 96)]
    # 2) 필터링 : 원본 몸무게 파운드 996~999는 지울 것
    total = total[(total['WEIGHT'] < 996)]
    # 3) BMI 값 처리
    total['HEIGHT(cm)'] = total['HEIGHT']*2.54
    total['WEIGHT(kg)'] = total['WEIGHT']*0.453592
    total.drop(['HEIGHT','WEIGHT'],axis=1,inplace=True)
    total['BMI'] = total['WEIGHT(kg)'] / ((total['HEIGHT(cm)']/100)**2)
    temp_bmi = total[['AGE','BMI','HEIGHT(cm)','WEIGHT(kg)']]

    ## 결측치 처리 : 결측 Feature 목록화 및 처리

    # ['ALCSTAT','CHLEV','CIGAREV2','CPLROU','EPHEV','HYPEV','HYPMED2','SMKSTAT2','TIRED_1']
    total['ALCSTAT'] = total['ALCSTAT'].apply(lambda x: x if x == 1.0 else 2)
    temp_list = ['ALCSTAT','CHLEV','CIGAREV2','CPLROU','EPHEV','HYPEV','HYPMED2','SMKSTAT2','TIRED_1']
    for i in temp_list:
        total[i] = total[i].apply(lambda x: x if x == 1.0 else 2)

    # 결측치 : 18세 이상 49세 이하는 응답, 그 외는 NaN
    total.loc[total['GENDER']==2.0,['AGE','GENDER','PREGNOW','PREGFLYR']]

    # 결측치 : 임신 관련 컬럼 2개 새로운 컬럼으로 합치기
    total['PREG'] = 1
    total.loc[(total['PREGNOW'] == 2)&(total['PREGFLYR'] == 2),'PREG'] = 2

    ##결측치 : 나머지 필요없는 컬럼 제거
    #* ['ECIGEV2','PIPEV2','SMKEV','SMKLSTB1','PREGFLYR','PREGNOW']
    total = total.drop(['ECIGEV2','PIPEV2','SMKEV','SMKLSTB1','PREGFLYR','PREGNOW'],axis=1)

    ## 3. 데이터 일원화 (2진 처리)
    # 'DIBEV1' 변경 : 2,7,8,9 → 0(아니오), 1,3 → 1 (예),  8은 데이터가 없음 
    total['DIBEV1'] = total['DIBEV1'].apply(lambda x: 1 if x in(1,3) else 0)
    # 0: 아니오, 1: 예로 통일함
    change_0 = ['ALCSTAT','ARTH1','CHLEV','CIGAREV2','CPLROU','EPHEV','HYPEV','HYPMDEV2','HYPMED2','INTIL2W','SMKSTAT2','TIRED_1','PREG']

    for i in change_0:
        total[i] = total[i].apply(lambda x: x if x == 1.0 else 0)
        
    # AGE 이상치 확인
    # 85세 이상은 85로 다 합쳐져있음 확인
    np.sort(total.AGE.unique()) ,total.AGE.value_counts(), total.GENDER.value_counts()

    # 성별 변경 : 1남자, 2여자  → 0남자, 1여자
    total.GENDER = total.GENDER - 1

    # 변경 : 1,2,7 → 0 (아니오), 3,9 → 1(예), 8은 데이터가 없음
    total['FSBALANC'] = total['FSBALANC'].apply(lambda x: 0 if x in(3,9) else 1)

    # 18세 이상 데이터만 가져오기
    total = total[total['AGE'] >= 18]

    ## 4. ID, TARGET 위치 선정
    # 'id' 컬럼 맨 앞으로 옮기기
    column_id = total.pop(total.columns[-4])
    total.insert(0, column_id.name, column_id)

    # Target 맨 뒤로 옮기기
    column_target = total.pop(total.columns[8])
    total.insert(24, column_target.name, column_target)

    # 인덱스 reset
    total = total.reset_index()
    total = total.drop(['index'],axis=1)

    # 5. 전처리 파일(CSV) 저장
    # 0,1로 이루어진 columns 리스트 
    countplot_list = ['ALCSTAT','ARTH1','CHLEV','CIGAREV2','CPLROU','EPHEV','HYPEV','HYPMDEV2','HYPMED2','INTIL2W','SMKSTAT2','TIRED_1','PREG']

    # for문을 이용해 당뇨병이 0과 1일때 각 컬럼의 비율을 나타낸 파이 그래프 생성
    for i in countplot_list:
        count_1 = total[total['DIBEV1'] == 1][i].value_counts().sort_index() # DIBEV1가 yes 일 때 리스트 안의 컬럼들 count
        count_0 = total[total['DIBEV1'] == 0][i].value_counts().sort_index() # DIBEV1가 no 일 때 리스트 안의 컬럼들 count
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].pie(count_1, labels=count_1.index, autopct='%1.1f%%', startangle=90, explode =(0,0.07), shadow=True, colors=('lightgreen','hotpink'))
        axes[0].set_title(f'{i} , diabetes : Yes')
        axes[1].pie(count_0, labels=count_0.index, autopct='%1.1f%%', startangle=90, explode =(0,0.07), shadow=True, colors=('lightgreen','hotpink'))
        axes[1].set_title(f'{i} , diabetes : No')
        plt.savefig(f'graph_diabetes_{i}.png')


    # 그래프 점검 통해 당뇨병과 설명력이 부족한 컬럼 제거
    total = total.drop(['CIGAREV2','CPLROU','SMKSTAT2','TIRED_1','PREG'],axis=1)
    total.to_csv("Data_preprocessing.csv", index=False)

    return None, 'Data_preprocessing'