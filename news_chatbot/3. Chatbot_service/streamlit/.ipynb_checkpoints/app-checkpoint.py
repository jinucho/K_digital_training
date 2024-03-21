import streamlit as st
from streamlit_chat import message
import json
import requests
import re

# Streamlit 앱의 메시지를 초기화합니다.
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 챗봇 타이틀을 설정합니다.
st.title("💬 뉴스챗봇")
st.text("Latest DB : 2024-03-06")


# 사용자가 질문을 입력할 수 있는 입력창을 생성합니다.
user_input = st.text_input("궁금한 뉴스를 물어보세요. :", key="input",value=None)

if user_input:
    st.session_state.messages.append({'text': user_input, 'is_user': True})
    with st.spinner('뉴스를 찾고 있어요!'):
    
        # url = 'http://127.0.0.1:8080/chat'  # 또는 실제 서버의 IP 주소를 사용
        url = 'http://localhost:8080/chat'  # Flask 서버의 URL
        # response = requests.post(url, data=user_input.encode('utf-8'))
        headers = {'Content-Type': 'application/json'}  # JSON 형식의 헤더 추가
        data = json.dumps({'user_input': user_input})  # JSON 형식으로 데이터 변환
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            bot_response = response.json()  # 또는 response.text, 응답 형식에 따라 달라질 수 있음
            combined_message = bot_response['result']  # 챗봇의 답변을 시작으로 합니다.
    
        # Document 객체의 URL과 내용을 추출하여 combined_message에 추가합니다.
            for i in range(len(bot_response['source_documents'])-1):
                if bot_response['source_documents'][i]['page_content']:
                    url_match = re.search('Link: (http[^\n]+)\nDescription', bot_response['source_documents'][i]['page_content'])
                    if url_match:
                        extracted_url = url_match.group(1)
                        combined_message += f"\n관련 링크{i}: [여기를 클릭하세요]({extracted_url})"
                        
            st.session_state.messages.append({'text': combined_message, 'is_user': False})


                      
# 이전 대화를 화면에 표시합니다.
for index, message_data in enumerate(st.session_state.messages):
    is_user = message_data['is_user']
    text = message_data['text']
    key = f"message-{index}"
    message(text, is_user=is_user, key=key)

