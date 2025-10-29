import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Serve index.html from root
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
        You are a binary AI-text detector.
        Respond with 'AI' if the text is AI-generated, or 'HUMAN' if written by a human.
        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a strict AI detector."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        verdict = response.choices[0].message.content.strip().upper()
        if "AI" in verdict:
            return jsonify({"result": "❌ FLAGGED: AI-GENERATED"})
        if "HUMAN" in verdict:
            return jsonify({"result": "✅ CLEAR: HUMAN-WRITTEN"})
        return jsonify({"result": f"⚠️ UNKNOWN RESPONSE: {verdict}"})
    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {str(e)}"}), 500


# Serve any other static files (CSS, JS, etc.)
@app.route("/<path:path>")
def serve_static_files(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
