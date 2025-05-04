# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for Streamlit
RUN apt-get update && apt-get install -y cmake g++ && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8502

# Command to start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8502", "--server.headless", "true"]