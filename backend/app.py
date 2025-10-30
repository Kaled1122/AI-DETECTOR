from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client safely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Root route (this is what fixes the “train has not arrived” issue)
@app.route("/")
def home():
    return jsonify({"message": "✅ AI Detector backend is running."})

# ✅ Detection route
@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided."}), 400

        # Call GPT model (example)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Detect if the text is AI-generated or human-written."},
                {"role": "user", "content": text},
            ],
        )

        result = completion.choices[0].message.content
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
