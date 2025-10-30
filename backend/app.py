import os
import json
import re
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# -----------------------------
# APP SETUP
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LOG_FILE = "detection_log.jsonl"

# -----------------------------
# HELPERS
# -----------------------------
def log_detection(entry: dict):
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def anonymize_text(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)
    text = re.sub(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", "[name]", text)
    text = re.sub(r"\+?\d[\d\s-]{7,}\d", "[number]", text)
    return text

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ CIPD AI Linguistic Detector backend is active.",
        "version": "2.1.0",
        "note": "Indicative linguistic analysis and humanization API."
    })


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    safe_text = anonymize_text(text)

    prompt = f"""
You are an impartial linguistic reviewer assisting an academic integrity check
in line with CIPD AI Policy v1.0.

Your task:
1. Examine the text below for stylistic, structural, and linguistic indicators
   that might suggest machine assistance (e.g., uniform tone, lack of hedging,
   over-formal transitions, repetitive sentence structure).
2. Estimate a **likelihood score (0–100)** for AI assistance.
3. Justify your reasoning neutrally and recommend a review level.

Return **only JSON** with these keys:
- "ai_confidence": integer 0–100 (estimated likelihood of AI assistance)
- "indicators": brief list of observed linguistic features
- "recommendation": "Low", "Medium", or "High" review priority
- "note": one-sentence reminder that this is *indicative only* and
  must be confirmed by human assessors.

Text for review:
{safe_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a neutral academic linguistic reviewer returning valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"(\d{1,3})", raw)
            parsed = {
                "ai_confidence": int(match.group(1)) if match else 50,
                "indicators": ["Parsing fallback"],
                "recommendation": "Medium",
                "note": "Indicative only; confirm manually."
            }

        ai_conf = max(0, min(100, int(parsed.get("ai_confidence", 50))))
        indicators = parsed.get("indicators", [])
        recommendation = parsed.get("recommendation", "Medium")
        note = parsed.get("note", "Indicative only; confirm manually.")

        if ai_conf >= 70:
            level = "High"
            color = "red"
        elif ai_conf >= 40:
            level = "Medium"
            color = "amber"
        else:
            level = "Low"
            color = "green"

        log_detection({
            "confidence": ai_conf,
            "level": level,
            "recommendation": recommendation,
            "indicators": indicators
        })

        return jsonify({
            "percentage": f"{ai_conf}%",
            "review_priority": level,
            "color": color,
            "indicators": indicators,
            "recommendation": recommendation,
            "note": note,
            "disclaimer": "This analysis is linguistic and indicative only; CIPD policy requires human verification."
        })

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500


@app.route("/humanize", methods=["POST"])
def humanize():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
Rewrite the following text to sound as naturally human as possible —
with varied sentence rhythm, natural pauses, and subtle emotional nuance —
while keeping meaning and professionalism intact.
Return only the rewritten text.

Text:
{text}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a linguistic stylist who rewrites text in a natural, human tone."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        rewritten = response.choices[0].message.content.strip()
        return jsonify({"humanized_text": rewritten})
    except Exception as e:
        return jsonify({"error": f"Humanization failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
