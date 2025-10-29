import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return jsonify({"message": "✅ AI Detector backend is running."})

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400

        # Strict binary detection instruction
        prompt = f"""
        You are a binary AI text detector.
        Analyze the text below and respond ONLY with one of the following:
        - "AI" if it is AI-generated.
        - "HUMAN" if it is human-written.
        No explanations. No extra words.

        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict AI detection engine."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        verdict = response.choices[0].message.content.strip().upper()

        # Format output for frontend
        if "AI" in verdict:
            result = "❌ FLAGGED: AI-GENERATED"
        else:
            result = "✅ CLEAR: HUMAN-WRITTEN"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
