import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Load your API key from Railway environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return jsonify({"message": "✅ AI Detector backend is running."})

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
        You are a binary AI-text detector.
        Analyze the text below and respond with one word only:
        - "AI" if it is AI-generated.
        - "HUMAN" if it is human-written.
        Text:
        {text}
        """

        # --- GPT-5 request ---
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
        # This ensures a readable JSON instead of a crash
        return jsonify({"error": f"OpenAI request failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
