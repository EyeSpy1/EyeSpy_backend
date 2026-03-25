FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install ffmpeg and build tools for other pip packages like 'av'
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dlib crucially via conda to get the PRE-COMPILED binary
# This completely bypasses the 8GB RAM compilation limit (Out Of Memory Error) on Render!
RUN conda install -y -c conda-forge python=3.11 dlib

# Copy requirements and install via pip
COPY requirements.txt .
RUN pip install --upgrade pip wheel packaging "setuptools<70.0.0"
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports for Flask and Streamlit
EXPOSE 8501 8502

# The entrypoint leverages supervisord to start both Flask and Streamlit simultaneously
CMD ["supervisord", "-c", "supervisord.conf"]
