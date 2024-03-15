# `Title: Be the information hub`</br>    
### team: ☕ espresso conpanna    </br>    
<li>Member : Cho Jinwu/ Haam EunGyu </li>
<li>Project Period : 2024-03-06 ~ 2024-03-14 </li>
## :books: skill
- **Programming** <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
- **Framework** <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/flask-412991?style=for-the-badge&logo=flask&logoColor=white">
- **Tools** <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/pycharm-000000?style=for-the-badge&logo=pycharm&logoColor=white"> <img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"> <img src="https://img.shields.io/badge/powerbi-E97627?style=for-the-badge&logo=powerbi&logoColor=white">
- **Git** <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-b
### `Project: Build Chatbot service with RAG.` </br>  
#### 1. Intoroduce Project</br>
<ol> 🌄 outlook
  <ol>
    <li>Gathering and preprocessing the data. </li>    
    <li>Compress the data to information. </li>   
    <li>RAG, Prepare information to embedding.</li>    
    <li>RAG, Connect LLM with information.</li>   
    <li>Build ML container.</li>   
    <li>Build serving system in streamlit, temporarily.</li>   
  </ol>
</ol>



<p align="center"><img src="https://github.com/jinucho/DL_for_pets/assets/151902373/6ce0d274-0387-49c2-a898-287427eb69c5" width="300" height="300"/>


# 딥러닝 프로젝트(강아지 안구 질병 예측)</br>Dog eye disease prediction
- Team : 숨참고딥다이브

## :sunny:팀구성 
  * 👥팀원 : 김유진, 송영달, 이수현, 이호, 조진우
  * :clock1:시작일 : 2024.01.29(월)
  * ⏰목표일 : 2024.02.08(목)

## 목차(INDEX)
&emsp;&ensp;Ⅰ. 🏁프로젝트 목적</br>
&emsp;&emsp;&emsp;- 반려견 안구 사진을 통한 질병 예측 서비스 개발</br>
&emsp;&ensp;Ⅱ. 📑데이터의 구성확인</br>
&emsp;&emsp;&emsp;1.- 출처 : [AIHub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=562)</br>
&emsp;&emsp;&emsp;2.- 데이터파일 구성  : 이미지 & json</br>
&emsp;&ensp;Ⅲ. 📑원본 데이터 분석</br>
&emsp;&emsp;&emsp;1. 질병 종류 확인</br>
&emsp;&emsp;&emsp; 결막염 / 궤양성각막질환 / 백내장 / 비궤양성각막질환 / 색소침착성각막염 / 안검내반증 / 안검염 / 안검종양 / 유루증 / 핵경화</br>
&emsp;&emsp;&emsp;2. 학습에 사용할 질병 별 이미지 및 json file 형식 확인</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DeepDive/assets/133849027/877b6c7e-9395-438c-a29e-51acc5158a14" width="600" height="300"></br>
&emsp;&ensp;Ⅳ. 📋데이터셋 구성</br>
&emsp;&emsp;&emsp;1. 이미지 및 json의 필요한 label만 추출하여 데이터셋 구성</br>
&emsp;&emsp;&emsp;2. 각 이미지와 label을 통해 질병(6종류)당 one-hot vector 1개를 target으로 지정</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DeepDive/assets/133849027/a0746514-d843-42dd-a583-82a7ab9379cc" width="600" height="300"></br>
&emsp;&ensp;Ⅴ. ✔안구질환 식별 프로세스</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DeepDive/assets/133849027/0c650aba-c906-40c0-b6dd-df0bb5258ef3" width="800" height="300"></br>
&emsp;&ensp;VI. ✔학습 모델과 모델 성능평가</br>
&emsp;&emsp;&emsp; AlexNet / VGG19 / ViT / ResNet50 / DenseNet201 / GoogleNet</br>
&emsp;&emsp;&emsp;1. AlexNet</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/7b10dc43-7501-4542-ab28-09d7be98d87b" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/71d14882-3010-4170-9f02-f46afe29b229" width="300"></br>
&emsp;&emsp;&emsp;2. VGG19</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/035a63ef-ebd8-48c3-9db1-b3f850af048c" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/15f9a183-a611-45b9-a757-f8c251e76803" width="300"></br>
&emsp;&emsp;&emsp;3. ViT</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/66013dc7-ef86-45cb-948d-4740a9ba1b7d" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/65ee3e47-4d85-4bac-bcaa-d1626a552c85" width="300"></br>
&emsp;&emsp;&emsp;4. ResNet50</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/fb318485-7cfa-4a4f-a30c-7f39017fd384" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/28292bac-55e9-4199-9a1d-258ac5e520f7" width="300"></br>
&emsp;&emsp;&emsp;5. DenseNet201</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/fdf059bd-63a6-4fcb-bf23-6209fe2dbed5" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/841aec41-586d-4833-bd39-e5c9de86cd46" width="300"></br>
&emsp;&emsp;&emsp;6. GoogleNet</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/d7ab901c-255b-471d-810a-a72cb179de0c" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/b5837a0f-90fa-4845-a2b2-debf2c213520" width="300"></br>
&emsp;&emsp;&emsp; Model 성능 비교</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/ba990145-71c4-49ee-8e46-63eacf050a31" width="300"> <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/e179f70b-4803-4995-9b8f-628d58207035" width="300"></br>
&emsp;&ensp;VII. 📑Streamlit 서비스</br>
&emsp;&emsp;&emsp; <img src="https://github.com/jinucho/DL_for_pets/assets/133849027/4c5eb98c-a4f6-47e0-b51e-2a673d03c3bd" width="700"></br>

&emsp;&ensp;VIII. 🚨사용기술</br>
![image](https://github.com/jinucho/DeepDive/assets/133849027/5c1ac76c-a1a7-41c8-8c8c-ccba2fcf1805)
