import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Force GPT-5 to return only one of two options
        prompt = f"""
        You are an AI detection classifier. 
        Analyze the following text carefully and determine if it was written by an AI or a human.

        Rules:
        - You must respond with EXACTLY one of these two labels:
          "AI-GENERATED" or "HUMAN-WRITTEN"
        - Do not explain or add extra text.

        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict AI detector. Always give a binary result."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        verdict = response.choices[0].message.content.strip().upper()

        if "AI" in verdict:
            flag = "❌ FLAGGED: AI-GENERATED"
        else:
            flag = "✅ CLEAR: HUMAN-WRITTEN"

        return jsonify({"result": flag})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
