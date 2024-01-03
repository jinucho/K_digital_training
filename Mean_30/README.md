# 🤖 머신러닝 프로젝트(당뇨병 예측모델)
Diabetes Predictive modeling
- Team : Mean_30


## 팀구성 :sunny:
  * 🏆팀명: Mean_30 (평균 30세)
  * 👥팀원 : 송영달, 송은민, 임창성, 조진우, 함은규
  * ⏰목표일 : 2024.01.05(금)

## index
  Ⅰ. 🏁프로젝트 목적  
    - 당뇨병을 예측모델을 개발 : 미국(NHIS) 자료 바탕  
  Ⅱ. 📑원본 데이터의 구성확인 (1차 분류)   
    - Datafile : CSV 5종 (csv, summary, layout, imputed incomes and paradata etc.  
  Ⅲ. 📑원본 데이터 분석 (2차 분류)  
    1. 원본데이터 분류   
    2. Column(이하:Feature) 필터링 (Feature Selection : Feature extraction)   
  Ⅳ. 📋데이터 전처리   
    1. Datafile 통합(merge)  
    2. 결측값(nan) 처리   
  Ⅴ. ✔학습 모델과 모델 성능평가  
  Ⅵ. 🚨프로젝트의 아쉬운 점 과 사용기술  
  Ⅶ. 📶자료출처  

## Ⅰ. 프로젝트 목적 : 머신러닝을 이용한 당뇨병 예측모델 개발
## Ⅰ.Purpose : Build a Predictive modeling of checking diabetes by using machine learning
  * 처리과정
  * Processing
    1) 적절한 전처리를 통한 데이터 추출
       filtering/arranging data by proper preprocessing
    2) 다양한 모델과 하이퍼파라미터를 조합(비교)을 통한 최적의 예측모델 구축
       Building an optimized model by compare within the variety models and within hyperparameters.
    - 기초자료 출처 : 미국(NHIS)
    - RawData resource : NHIS
    - Url : https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm

## Ⅱ. 원본 데이터의 구성확인 (1차 분류) 
* 📑 기초 데이터
  1. Datafile : CSV 5종
      1) Family file
      2) Household file
      3) Person file
      4) Sample Child file
      5) Sample Adult file
      - Datafile summary               [Datafile Feature 설명]
      - Datafile layout                   [Datafile Feature 상세 설명 및 질문]

  2. imputed incomes                  [Data파일]  *사용불가
  3. Paradata                             [CSV 파일]  *불필요 (설문관련)
  4. Functioning and Disability       [PDF 파일]  *불필요 (PDF)

## Ⅲ. 원본 데이터 분석 (2차 분류)
  1. 원본 데이터 분류
    1) 분류 기준 : 당뇨병 관계성(논문 등 참조) 및 결측 값이 적은 항목
     예) 류마티즘 약 복용 등
    2) 분류 방법 : 데이터 구분(사용, 점검, 참고)
  2. Feature 필터링 (Feature Selection : Feature extraction)   

## Ⅳ. 데이터 전처리
  1. Datafile 통합(merge) 
    1) 3개 데이터셋 : Family , Sample Child, Sample Adult
    2) 발생문제
       - 현      상 : 통합시, 동일 Feature가 있음에도 Feature이 새로 생성됨
       - 해결방법 : combine_first 함수 사용 
  2. 전처리/결측치(nan) 처리
  3. Feature 선정(1): 논문 참조
  4. Feature 선정(2): 인과관계 확인 필요항목들 추가
    1) 흡연여부 
    2) 성별, 임신여부(임신성 당뇨)
    3) 인종별 차이(식습관)
  5. 1차 전처리 데이터 학습모델 성능 (preview)
  6. 2차 전처리 : Feature 추가점검 (당뇨 여부에 미치는 영향 점검)

## Ⅴ. 학습 모델과 모델 성능평가 :point_left:
  1. 🌱import 라이브러리 
  2. 🌱수치형 데이터 스케일링
  3. 🌿onehotencoding (2진값 外)
  4. 🌿AutoML(Pycaret) 최적화 모델링
  5. 🪴모델 학습검증
  6. 🌲Threshold 값 추정
  7. 🍎Test Value : 모델 성능 측정

    
## Ⅵ. 프로젝트의 아쉬운 점과 사용기술
  1. 😭최적화의 한계 : presision 0.32 max 
    1) 정밀도(Precision 값)이 더이상 줄어들지 않음
    2) 💪시도  :  Validate까지 0.7~0.8로 준수 → 최종 Test에서 0.30~0.32로 형성
        - 컬럼 재정리
        - 모델 변경과 하이퍼파라미터를 변경 
        - 샘플링 방법 변경
    3) :collision:결과 : 결과값에 변동이 없음, 

  2. 대안: Feature select부터 다시 Bulid해야 할 것으로 사료
    - 프로젝트 목표일로 이 값을 인정 (또한 다른 참조문서는 recall이 0.11 수준)

  3. 마무리 :pray:
    1) 전처리 및 그래프, 그리고 머신러닝까지 아우룰 수 있는 좋을 기회였음
    2) 팀원들과 논의하고 기술을 공유하고 향상시킬 수 있는 기간이었음

  4. 사용기술
    - python, pandas, numpy, matplotlib, sklearn and pycaret etc.

  5. 💬자료출처 
   1) 기초데이터
       - https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm

   2) 기술자료
       - 월간당뇨 11월호 :  소아 당뇨병이란 무엇인가? 
       - 연세대학교 의과대학 소아과학교실 : 소아연령에서의 2형 당뇨병의 임상적 고찰

   3) 세계 당뇨지도 
       - https://www.visualcapitalist.com/cp/diabetes-rates-by-country/. 


## Ⅶ. 자료출처
  1. 기초데이터 :  https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm
  2. 참조기술자료
    1) 월간당뇨 11월호 :  소아 당뇨병이란 무엇인가?
    2) 연세대학교 의과대학 소아과학교실 : 소아연령에서의 2형 당뇨병의 임상적 고찰
  3. 세계당뇨 지도 
    https://www.visualcapitalist.com/cp/diabetes-rates-by-country/. 
  4. Used program skills : python, pandas, numpy, matplotlib, sklearn and pycaret etc.




