import os

import streamlit as st
import requests
import base64
from PIL import Image
import time

if __name__ == '__main__':
    try:
        # 첫 번째 페이지 - 반려견 정보 입력
        st.set_page_config(layout="wide", page_title="반려견 안구질환 진단 PY1")


        with st.sidebar:
            st.write("## 페이지 네비게이션")
            # 사이드바 라디오 버튼의 선택이 st.session_state를 참조하도록 설정
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "반려견 안구진단 인트로"  # 초기 상태 설정
            st.session_state.selected_page = st.radio("페이지 선택", ["반려견 안구진단 인트로", "반려견 정보 입력", "반려견 안구 사진 진단"],
                                                      index=["반려견 안구진단 인트로", "반려견 정보 입력", "반려견 안구 사진 진단"].index(st.session_state.selected_page))

        if st.session_state.selected_page == "반려견 안구진단 인트로":
            st.markdown("<h2 style='color: blue; font-size: 32px;'>반려견 안구질환 진단 딥러닝 프로젝트</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:dark gray; font-size: 27px;'>Team: 숨참고 딥다이브!</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:gray; font-size: 15px;'>Member: 조진우, 이수현, 이호, 송영달, 김유진</h2>",
                        unsafe_allow_html=True)

            # 메인 페이지 강아지 이미지 추가
            image = Image.open("st\pngtree_dog.png")  # 강아지 아이콘 이미지 파일 경로 지정
            st.image(image, use_column_width=True)

            next_button_clicked = st.button("우리 강아지 안구질환 진단 시작💊🧪")
            if next_button_clicked:
                st.session_state.selected_page = "반려견 정보 입력"  # 페이지 이동

        # 페이지 2: 반려견 정보 입력
        elif st.session_state.selected_page == "반려견 정보 입력":
            st.markdown("<h2 style='color: blue; font-size: 32px;'>우리 강아지 눈 건강 확인</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: dark gray; font-size: 26px;'>반려견 정보를 입력하세요</h2>", unsafe_allow_html=True)

            # st.title("반려견 정보 입력")
            st.markdown("<h2 style='color: orange; font-size: 20px;'>반려견 정보 입력</h2>", unsafe_allow_html=True)
            # 기존 입력값을 불러옵니다 (있을 경우)
            pet_name = st.text_input("반려견 이름", value=st.session_state.get('pet_name', ''))
            pet_age = st.number_input("반려견 나이", 0, 30, value=st.session_state.get('pet_age', 0))
            pet_breed = st.text_input("반려견 품종", value=st.session_state.get('pet_breed', ''))
            pet_weight = st.number_input("반려견 몸무게 (kg)", 0.0, 100.0, value=st.session_state.get('pet_weight', 0.0))

            # 저장 버튼
            if st.button("입력완료"):
                st.session_state.pet_name = pet_name
                st.session_state.pet_age = pet_age
                st.session_state.pet_breed = pet_breed
                st.session_state.pet_weight = pet_weight
                # st.success("반려견 정보가 저장되었습니다")

                # 저장된 정보 출력
                # with st.expander("우리 강아지 정보:"):
                # st.write("우리 강아지 입력 정보")
                st.markdown("<h2 style='color: orange; font-size: 20px;'>우리 강아지 입력 정보</h2>", unsafe_allow_html=True)
                st.write('👀이름: ', pet_name)
                st.write('🦌나이(세): ', pet_age)
                st.write('🐾몸무게(kg): ', pet_weight)

            next_button_clicked1 = st.button("다음")

            if next_button_clicked1:
                st.session_state.selected_page = "반려견 안구 사진 진단"  # 페이지 이동


        # 페이지 3: 반려견 안구 사진 진단
        elif st.session_state.selected_page == "반려견 안구 사진 진단":
            st.markdown("<h2 style='color: blue; font-size: 28px;'>우리 강아지 안구 질환 진단하기</h2>", unsafe_allow_html=True)
            st.markdown("<h2 style='color: dark gray; font-size: 24px;'> 사진 한장으로 가능한 안구질환을 감지해보세요</h2>",
                        unsafe_allow_html=True)

            MAX_FILE_SIZE = 5 * 1024 * 1024

            my_upload = st.file_uploader("사진 업로드", type=["png", "jpg"])

            if my_upload is not None:
                if my_upload.size > MAX_FILE_SIZE:
                    st.error("사진 크기는 5MB 이하로 설정하세요.")
                else:
                    st.image(my_upload, width=250)

                    # with st.expander("우리 강아지 안구질환 진단하기"):
                    button_clicked = st.button("진단 시작")


                    if button_clicked:
                        with st.spinner('진단을 수행 중입니다...'):
                            img_bytes = my_upload.read()
                            encoded = base64.b64encode(img_bytes).decode('utf-8')  # byte를 문자열로 변환

                            url = 'http://localhost:5002/predict'  # 또는 실제 서버의 IP 주소를 사용
                            data = {"image_data": encoded}
                            response = requests.post(url, data=data)

                            if response.ok:
                                yolo_pic_folder_path = 'yolo_pic'
                                file_list = os.listdir(yolo_pic_folder_path)
                                yolo_image_files = [file for file in file_list if
                                               file.lower().endswith(('.png', '.jpg', '.jpeg'))]
                                # for image_file in yolo_image_files:
                                #
                                #     image_path = os.path.join(yolo_pic_folder_path, image_file)
                                #     image = Image.open(image_path)
                                #     st.image(image, width=image.width * 20, use_column_width=False)
                                for i in range(0, len(yolo_image_files), 2):
                                    cols = st.columns(len(yolo_image_files))
                                    for j in range(2):
                                        if i + j < len(yolo_image_files):
                                            with cols[j]:
                                                image_file = yolo_image_files[i + j]
                                                image_path = os.path.join(yolo_pic_folder_path, image_file)
                                                image = Image.open(image_path)
                                                st.image(image, width=300)

                                de_pic_folder_path = 'de_pic'
                                de_file_list = os.listdir(de_pic_folder_path)
                                de_image_files = [file for file in de_file_list if
                                               file.lower().endswith(('.png', '.jpg', '.jpeg'))]

                                jso = response.json()
                                j_sp = jso.split('_')

                                for i in range(0,6):
                                    image_path = os.path.join(de_pic_folder_path, de_image_files[i])
                                    st.image(image_path, width=300, use_column_width=False)
                                    st.write(j_sp[i])

                            else:
                                st.write('서버 오류로 인해 분석을 완료할 수 없습니다')
    #
    #
    except Exception as e:
        st.write(f"오류가 발생하였습니다:{str(e)}")