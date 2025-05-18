FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    build-essential cmake && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        streamlit \
        streamlit-webrtc \
        opencv-python-headless \
        dlib \
        numpy \
        scipy \
        av \
        pydub

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]