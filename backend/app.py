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

# ⚡ Load a small open model for local probability stats
# (you can swap for any causal LM like "mistralai/Mistral-7B-Instruct-v0.2"
# if you have a GPU)
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
    return re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", text)


def log_result(entry):
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def compute_perplexity_entropy(text):
    """True perplexity + entropy from token-level logprobs."""
    enc = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        logits = outputs.logits

    # Perplexity
    ppl = math.exp(loss.item())

    # Entropy
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)) / probs.numel()
    entropy = entropy.item()

    return ppl, entropy


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ CIPD AI Detector (True Statistical Hybrid) is active.",
        "local_model": MODEL_NAME,
        "version": "4.0.0"
    })


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    safe_text = anonymize(text)

    try:
        # --- Step 1: Statistical analysis ---
        perplexity, entropy = compute_perplexity_entropy(safe_text)

        # Normalize to 0–100 scale
        # (lower perplexity = more AI-like, lower entropy = more AI-like)
        norm_ppl = max(0, min(100, 100 - (min(perplexity, 200) / 2)))
        norm_entropy = max(0, min(100, 100 - entropy * 20))

        # --- Step 2: Linguistic GPT-5 analysis ---
        prompt = f"""
        You are a linguistic integrity reviewer following CIPD AI Policy v2.0.
        Examine the text for AI-like linguistic indicators and return JSON only:
        {{
          "ai_score": integer 0–100,
          "confidence_level": "Low"|"Medium"|"High",
          "indicators": ["brief list of observed traits"],
          "assessment": "short neutral explanation"
        }}
        Text:
        {safe_text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a neutral linguistic auditor returning pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"ai_score": 50, "confidence_level": "Medium",
                      "indicators": ["Parsing fallback"], "assessment": "Output parse failed."}

        ai_score = int(parsed.get("ai_score", 50))
        indicators = parsed.get("indicators", [])
        assessment = parsed.get("assessment", "")
        level = parsed.get("confidence_level", "Medium")

        # --- Step 3: Combine all three metrics ---
        combined = int((ai_score * 0.6) + (norm_ppl * 0.25) + (norm_entropy * 0.15))

        if combined < 40:
            color, label = "green", "Low likelihood of AI authorship"
        elif combined < 70:
            color, label = "orange", "Possible AI indicators"
        else:
            color, label = "red", "High likelihood of AI assistance"

        result = {
            "percentage": f"{combined}%",
            "ai_score": ai_score,
            "perplexity": round(perplexity, 2),
            "entropy": round(entropy, 3),
            "normalized_perplexity": norm_ppl,
            "normalized_entropy": norm_entropy,
            "review_priority": label,
            "color": color,
            "reason": assessment,
            "indicators": ", ".join(indicators),
            "recommendation": level,
            "note": (
                "This hybrid score combines linguistic, statistical, and entropy analyses. "
                "It is indicative only and must be contextually reviewed by human assessors."
            )
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
        Rewrite the text naturally in human rhythm, varying tone and sentence length.
        Preserve meaning but reduce structural regularity and artificial flow.
        Return only the rewritten text.
        Text:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a skilled editor making text sound genuinely human."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
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
