import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# -----------------------------
# APP SETUP
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
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
        You are an AI text detector. Return one word only:
        - "AI" if the text is AI-generated.
        - "HUMAN" if the text is written by a human.
        Text: {text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict AI detector."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        verdict = (response.choices[0].message.content or "").strip().upper()
        result = "❌ FLAGGED: AI-GENERATED" if "AI" in verdict else "✅ CLEAR: HUMAN-WRITTEN"
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
