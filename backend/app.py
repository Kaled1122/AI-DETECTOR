import os, json, re, math, datetime, torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

LOG_FILE = "detection_log.jsonl"

# Lightweight local LM for perplexity / entropy metrics
MODEL_NAME = os.getenv("LOCAL_MODEL", "distilgpt2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def anonymize(text):
    """Remove sensitive info like emails."""
    return re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)

def log_result(entry):
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def compute_perplexity_entropy(text):
    """Compute perplexity and entropy from token log probabilities."""
    enc = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        logits = outputs.logits
    ppl = math.exp(loss.item())
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)) / probs.numel()
    return ppl, entropy.item()

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ CIPD Linguistic Authenticity Analyzer running.",
        "local_model": MODEL_NAME,
        "version": "5.1.1 (No-temp)"
    })

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        safe_text = anonymize(text)

        # --- Step 1: Statistical metrics ---
        perplexity, entropy = compute_perplexity_entropy(safe_text)
        norm_ppl = max(0, min(100, 100 - (min(perplexity, 200) / 2)))
        norm_entropy = max(0, min(100, 100 - entropy * 20))

        # --- Step 2: Linguistic GPT-5 analysis ---
        prompt = f"""
        You are a linguistic authenticity reviewer under CIPD Policy v3.0.
        Examine this text for human-like or AI-like linguistic traits.
        Return ONLY valid JSON with:
        {{
          "ai_score": integer 0–100,
          "traits": ["short list of observed writing traits"],
          "summary": "one-sentence neutral observation"
        }}
        Text:
        {safe_text}
        """
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"ai_score": 50, "traits": ["Parsing fallback"], "summary": "No detailed linguistic analysis."}

        ai_score = int(parsed.get("ai_score", 50))
        traits = ", ".join(parsed.get("traits", []))
        summary = parsed.get("summary", "")

        # --- Step 3: Combine metrics ---
        combined = int((ai_score * 0.6) + (norm_ppl * 0.25) + (norm_entropy * 0.15))
        if combined < 40:
            verdict, color = "Looks Human", "green"
        elif combined < 70:
            verdict, color = "Mixed Traits", "orange"
        else:
            verdict, color = "Looks AI-Assisted", "red"

        result = {
            "authenticity_index": f"{combined}%",
            "verdict": verdict,
            "ai_score": ai_score,
            "language_flow": round(perplexity, 2),
            "variation": round(entropy, 3),
            "traits": traits,
            "summary": summary,
            "color": color,
            "note": (
                "This analysis estimates linguistic authenticity using statistical "
                "and stylistic features. It is indicative only and should be reviewed contextually."
            ),
            "weights": {
                "linguistic": 60,
                "statistical": 25,
                "entropy": 15
            }
        }

        log_result(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

@app.route("/humanize", methods=["POST"])
def humanize():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        prompt = f"""
        Rewrite the text with natural human rhythm and tone.
        Vary sentence length and phrasing while preserving meaning.
        Avoid robotic flow. Return only the rewritten version.
        Text:
        {text}
        """
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a skilled human editor."},
                {"role": "user", "content": prompt}
            ]
        )
        humanized = response.choices[0].message.content.strip()
        return jsonify({"humanized": humanized})

    except Exception as e:
        return jsonify({"error": f"Humanization failed: {str(e)}"}), 500

# ---------------------------------------------------------
# ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
