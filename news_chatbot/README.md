# `Title: Be the information hub`</br>    
### team: ☕ espresso conpanna    </br>    
<li>Member : Cho Jinwu/ Haam EunGyu </li>
<li>Project Period : 2024-02-27 ~ 2024-03-14 </li>

## :books: Used skill
- **Programming** <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
- **Library** <img src="https://img.shields.io/badge/scikitlearn-blue?style=for-the-badge&logo=scikitlearn&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-blue?style=for-the-badge&logo=Numpy&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-blue?style=for-the-badge&logo=Pandas&logoColor=white"> <img src="https://img.shields.io/badge/HUGGINGFACE-yellow?style=for-the-badge&logo=huggingface&logoColor=white"> <img src="https://img.shields.io/badge/Langchain-green?style=for-the-badge&logo=langchain&logoColor=white">  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/flask-412991?style=for-the-badge&logo=flask&logoColor=white">      
- **Environment** <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"> <img src="https://img.shields.io/badge/runpod-412991?style=for-the-badge&logo=Runpod&logoColor=white"> <img src="https://img.shields.io/badge/Docker-412991?style=for-the-badge&logo=Docker&logoColor=white">


## 목차(INDEX)
&emsp;&ensp;Ⅰ. 주제 선정</br>&emsp;&ensp;Ⅱ. Data preprocessing & Modeling</br>&emsp;&ensp;Ⅲ. Chatbot 구현</br>&emsp;&ensp;Ⅳ. 마무리</br>

## Ⅰ. 주제선정
  **1. 대화형 AI Chatbot 서비스 구현**</br>
       &nbsp;&nbsp;&nbsp; 1) 뉴스기사 요약 --> 최종 drop 결정</br>
       &nbsp;&nbsp;&nbsp; 2) 뉴스기사 기반 챗봇 서비스 구현(w/ RAG)</br>
       
  **2. 자료출처**</br>
       &nbsp;&nbsp;&nbsp; 1) [AIHUB 문서 요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=97)
       &nbsp;&nbsp;&nbsp; 2) [AIHUB 뉴스기사 기계독해](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577)

## Ⅱ. Data preprocessing & Modeling
**1. 데이터 전처리**</br>
       &nbsp;&nbsp;&nbsp; 1) Json 기반 데이터 -> 데이터프레임으로 필요한 부분만 추출</br>
       &nbsp;&nbsp;&nbsp; 2) 학습에 필요한 형태로 재정리</br>
       &nbsp;&nbsp;&nbsp; 3) Tokenizer</br>
       
**2. 모델 학습**</br>
       &nbsp;&nbsp;&nbsp; 1) 요약모델 : KoBertsum / Kobart / Bertsum</br>
       &nbsp;&nbsp;&nbsp; 2) 챗봇모델 : Ko-platyi-6b / KoLLama2-7b / KoRMKV / KULLM / Llama-7b / OpenAI API --> 모델 학습 실패로 OpenAI API를 이용</br>
       &nbsp;&nbsp;&nbsp; 3) RAG(Langchain library 사용)</br>

## Ⅲ. Chatbot 구현
**1. 챗봇 서비스**</br>
       &nbsp;&nbsp;&nbsp; 1) Streamlit / Flask 사용</br>
       &nbsp;&nbsp;&nbsp; 2) Docker를 이용한 Container화</br>
       &nbsp;&nbsp;&nbsp; 3) 웹 배포</br>

## Ⅳ. 마무리
**1. 챗봇 서비스 구현**</br>
       &nbsp;&nbsp;&nbsp; <img src="https://github.com/jinucho/espresso_conpanna/assets/133849027/7e05ca29-5e91-4218-a1c9-b092e5a7074a" width="300"> </br>
**2. 후기**</br>
       &nbsp;&nbsp;&nbsp; 1) LLM에서 학습된 정보에 한정되지 않고, RAG 방법을 사용함으로써 새로운 정보를 얻을 수 있다는 것을 알게 되었습니다.</br>
       &nbsp;&nbsp;&nbsp; 2) 우선순위 선정 미흡으로 목표한 100% 서비스화 구축에 실패(local model 사용 불가 / OpenAI API 사용)</br>
       &nbsp;&nbsp;&nbsp; 3) 챗봇 모델 학습 단계에서 모델에 대한 분석이 부족하여 잘 못된 방향으로 학습을 진행(영어 기반 모델에 한글자료로 LoRA 튜닝)</br>
