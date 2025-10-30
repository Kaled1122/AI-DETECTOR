import os
import json
import re
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
    return jsonify({"message": "✅ AI Detector backend is running with percentage mode."})


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
        You are an AI text detection system.
        Analyze the following text and respond **only** in JSON with:
        - "ai_confidence": integer from 0–100 showing probability it was AI-generated
        - "reason": a short one-sentence justification
        Text:
        {text}
        """

        # GPT-5 or GPT-4o-mini (faster)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI content detector that returns clean JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        # --- Try parsing JSON safely ---
        try:
            parsed = json.loads(raw_output)
        except Exception:
            # Fallback: extract digits if not valid JSON
            match = re.search(r"(\d{1,3})", raw_output)
            percent = int(match.group(1)) if match else 50
            parsed = {"ai_confidence": percent, "reason": "Could not parse structured output."}

        ai_conf = int(parsed.get("ai_confidence", 50))
        reason = parsed.get("reason", "No explanation provided.")

        verdict = "❌ AI-Generated" if ai_conf >= 50 else "✅ Human-Written"

        return jsonify({
            "confidence": f"{ai_conf}%",
            "result": verdict,
            "reason": reason
        })

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


# -----------------------------
# APP ENTRY
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
