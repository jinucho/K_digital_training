# 기본 이미지
FROM python:3.10.13-slim
WORKDIR /app

# flask 디렉토리의 파일들을 복사
COPY ./flask/* ./

# faiss_index 디렉토리도 복사
COPY ./flask/faiss_index ./faiss_index

# 필요한 패키지들 설치
RUN pip install --no-cache-dir -r requirements.txt

# flask_app.py 실행
CMD ["python", "flask_app.py"]


EXPOSE 8080
