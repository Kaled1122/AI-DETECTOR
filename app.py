import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

# -----------------------------
# SETUP
# -----------------------------
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# SERVE FRONTEND (index.html in root)
# -----------------------------
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

# -----------------------------
# AI DETECTION ENDPOINT
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # GPT-5 prompt
        prompt = f"""
        You are a binary AI-text detector.
        Analyze the text below and respond with one word only:
        - "AI" if it is AI-generated.
        - "HUMAN" if it is human-written.
        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict AI detector."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        verdict = (response.choices[0].message.content or "").strip().upper()

        if "AI" in verdict:
            result = "❌ FLAGGED: AI-GENERATED"
        elif "HUMAN" in verdict:
            result = "✅ CLEAR: HUMAN-WRITTEN"
        else:
            result = "⚠️ UNKNOWN RESPONSE"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {str(e)}"}), 500


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
