from flask import Flask, jsonify, request, Response
import psutil
import requests as req
from flask_cors import CORS

app = Flask(__name__)

# ✅ Allow all origins (your deployed frontend + localhost dev)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def home():
    return jsonify({"message": "Welcome to the EyeSpy Flask server!"})


@app.route("/start-detection", methods=["POST"])
def start_detection():
    """
    Streamlit is already running via supervisord on port 8502 internally.
    We just return the proxied public URL — /streamlit/ on this same server.
    """
    return jsonify({
        "message": "Detection started",
        "url": "https://eyespy-backend.onrender.com/streamlit/"
    }), 200


@app.route("/stop-detection", methods=["POST"])
def stop_detection():
    """Stop the Streamlit process managed by supervisord."""
    stopped = False
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "streamlit" in cmdline and "app.py" in cmdline:
                proc.kill()
                stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return jsonify({
        "message": "Detection stopped" if stopped else "No active detection found"
    }), 200


# ✅ Proxy ALL Streamlit traffic through Flask on the same public port
@app.route("/streamlit/", defaults={"path": ""})
@app.route("/streamlit/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def proxy_streamlit(path):
    """
    Render only exposes one port publicly (8501 = Flask).
    Streamlit runs internally on 8502.
    This proxy forwards all browser requests to Streamlit and returns the response.
    """
    target_url = f"http://localhost:8502/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode()}"

    try:
        resp = req.request(
            method=request.method,
            url=target_url,
            headers={
                k: v for k, v in request.headers
                if k.lower() not in ("host", "content-length")
            },
            data=request.get_data(),
            allow_redirects=False,
            timeout=15
        )

        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
        return Response(resp.content, resp.status_code, headers)

    except req.exceptions.ConnectionError:
        return jsonify({
            "error": "Streamlit is starting up. Please wait a few seconds and try again."
        }), 503


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)