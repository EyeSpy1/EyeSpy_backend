FROM python:3.11-slim

# Install system dependencies
# cmake and build-essential are required for dlib
# ffmpeg is required for pydub to process audio
# openblas and lapack are recommended for dlib performance
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Create unbuffered environment variable for python
ENV PYTHONUNBUFFERED=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports for Flask and Streamlit
EXPOSE 8501 8502

# The entrypoint leverages supervisord to start both Flask and Streamlit using the provided supervisord.conf
CMD ["supervisord", "-c", "supervisord.conf"]
