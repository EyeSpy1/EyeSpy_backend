from flask import Flask, jsonify, request
import subprocess
import threading
from flask_cors import CORS

app = Flask(__name__)

# Allow CORS from localhost with specific React port (5173 by default for Vite)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Start Streamlit app in a separate thread
def start_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py"])

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the EyeSpy Flask server!"})

@app.route("/start-detection", methods=["POST"])
def start_detection():
    """Starts the Streamlit app."""
    threading.Thread(target=start_streamlit).start()
    return jsonify({"message": "Streamlit app started"}), 200

@app.route("/stop-detection", methods=["POST"])
def stop_detection():
    """Stops the Streamlit app."""
    subprocess.Popen(["pkill", "-f", "streamlit"])
    return jsonify({"message": "Streamlit app stopped"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
