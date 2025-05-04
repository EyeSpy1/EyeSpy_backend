# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for dlib and supervisor
RUN apt-get update && apt-get install -y cmake g++ supervisor && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the ports for Flask and Streamlit
EXPOSE 8501
EXPOSE 8502

# Command to start supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]