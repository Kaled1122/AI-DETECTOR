import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

# Initialize Flask
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# FRONTEND: Serve index.html
# ---------------------------
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# ---------------------------
# API: AI Detection
# ---------------------------
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Prompt to GPT-5
        prompt = f"""
        You are an AI-text detector.
        If the following text is written by AI, respond only with "AI".
        If it is written by a human, respond only with "HUMAN".

        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict binary AI detector."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        verdict = (response.choices[0].message.content or "").strip().upper()

        if "AI" in verdict:
            return jsonify({"result": "❌ FLAGGED: AI-GENERATED"})
        elif "HUMAN" in verdict:
            return jsonify({"result": "✅ CLEAR: HUMAN-WRITTEN"})
        else:
            return jsonify({"result": f"⚠️ UNKNOWN RESPONSE: {verdict}"})
    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {str(e)}"}), 500


# ---------------------------
# Serve static assets (CSS/JS if any)
# ---------------------------
@app.route("/<path:path>")
def static_files(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
