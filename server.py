from flask import Flask, jsonify, request
import subprocess
import threading
import psutil  #
from flask_cors import CORS

app = Flask(__name__)

# Allow CORS from localhost with specific React port (5173 by default for Vite)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})


@app.route("/")
def home():
    return jsonify({"message": "Welcome to the EyeSpy Flask server!"})

@app.route("/start-detection", methods=["POST"])
def start_detection():
    """Returns the Streamlit app URL."""
    # Since supervisor handles starting Streamlit, it is already running.
    host = request.host.split(":")[0]
    streamlit_url = f"http://{host}:8502/"  # Assuming default Streamlit port mapped in Docker
    return jsonify({
        "message": "Streamlit app is running concurrently via initial setup", 
        "url": streamlit_url
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
