"""
Minimal API server exposing Carly over HTTP, for the web UI to call.

Run with: python3 api_server.py
Then open index.html in a browser (or serve it) -- it POSTs to
http://localhost:5001/ask

Requires: pip install flask flask-cors
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from generate_answer import generate_answer

app = Flask(__name__)
CORS(app)  # allows index.html (opened as a local file / different origin) to call this API

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
    app.run(port=5001, debug=True)