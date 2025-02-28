from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
# 기타 필요한 라이브러리 및 설정들을 여기에 포함시킵니다.

app = Flask(__name__)

load_dotenv("config.toml")


os.environ['OPENAI_API_KEY'] = os.getenv("openai_key")


embedding_model = OpenAIEmbeddings()

db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

rt_docsearch = db.as_retriever(search_type='similarity',search_kwargs={'k':3,'fetch_k':10})

openai = ChatOpenAI(
    model_name = 'gpt-3.5-turbo',
    # streaming=True, callbacks= [StreamingStdOutCallbackHandler()],
    temperature=0)

prompt_template = '''
    "{question}에 대해 본문 내용을 설명해줘.\n\n"
    "### 본문:\n{context}\n\n 응답:\n"
'''

prompt = PromptTemplate.from_template(prompt_template)


qa = RetrievalQA.from_chain_type(llm =openai,
                                 chain_type = 'stuff',
                                 retriever = rt_docsearch,
                                 return_source_documents=True,
                                chain_type_kwargs={'prompt':prompt})

@app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json['user_input']
#     print(user_input)
    
#     # 챗봇 응답 생성 로직
#     # bot_response = ...
#     bot_response = qa(user_input)
#     print(bot_response)
#     return jsonify({'response': bot_response})
def chat():
    user_input = request.json['user_input']
    print(user_input)
    
    # 챗봇 응답 생성
    bot_response = qa(user_input)

    # bot_response의 결과를 JSON 직렬화 가능한 형태로 변환
    response_data = {
        'query': bot_response['query'],
        'result': bot_response['result'],
        'source_documents': [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in bot_response['source_documents']
        ]
    }

    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080) # 서버에있는 모든 ip 주소의 5000번 포트로 접속가능
# http://192.168.10.100:8080/cha