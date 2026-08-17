"""
Minimal API server exposing Carly over HTTP, for the web UI to call.

Run with: python3 api_server.py
Then open index.html in a browser (or serve it) -- it POSTs to
http://localhost:5001/ask

Requires: pip install flask flask-cors
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from generate_answer import generate_answer
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ASSETS_DIR = ROOT / "assets"

app = Flask(__name__)
CORS(app)  # allows the web UI to call this API

# Serve the web UI (scripts/index.html) at the site root and under index.html
@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(str(SCRIPTS_DIR), "index.html")

# Serve assets (images/gifs) at /assets/
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(str(ASSETS_DIR), filename)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id") or "default"

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        answer = generate_answer(question, session_id=session_id)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    # bind to 0.0.0.0 so the service is reachable from outside the container/host
    app.run(host="0.0.0.0", port=port, debug=(os.environ.get("FLASK_DEBUG", "false").lower() == "true"))