from flask import Flask, jsonify, request
import subprocess
import threading
import psutil  #
from flask_cors import CORS

app = Flask(__name__)

# Allow CORS from localhost with specific React port (5173 by default for Vite)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})


def start_streamlit():
    # Check if Streamlit is already running
    for process in psutil.process_iter(['pid', 'name']):
        if 'streamlit' in process.info['name']:
            print("Streamlit is already running.")
            return  # Exit the function if Streamlit is already running

    # Start Streamlit on port 8502
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8502"])
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the EyeSpy Flask server!"})


@app.route("/start-detection", methods=["POST"])
@app.route("/start-detection", methods=["POST"])
def start_detection():
    """Starts the Streamlit app."""
    # Return the Streamlit app's public URL
    return jsonify({"message": "Streamlit app started", "url": "https://drowsiness-streamlit.onrender.com"}), 200
@app.route("/stop-detection", methods=["POST"])
def stop_detection():
    """Stops the Streamlit app."""
    subprocess.Popen(["pkill", "-f", "streamlit"])
    return jsonify({"message": "Streamlit app stopped"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
