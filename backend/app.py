import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
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
        You are an AI-text detector.
        If the text is AI-generated, reply 'AI'.
        If human-written, reply 'HUMAN'.
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
            result = "❌ FLAGGED: AI-GENERATED"
        elif "HUMAN" in verdict:
            result = "✅ CLEAR: HUMAN-WRITTEN"
        else:
            result = f"⚠️ UNKNOWN RESPONSE: {verdict}"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
