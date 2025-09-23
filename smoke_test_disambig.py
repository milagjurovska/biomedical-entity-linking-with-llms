import os, json, re, requests

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_cui": {"type": "string"},
        "selected_name": {"type": "string"},
        "evidence_span": {"type": ["string", "null"]},
    },
    "required": ["selected_cui", "selected_name"],
    "additionalProperties": False,
}

def disambiguate(mention, left_ctx, right_ctx, candidates):
    prompt = f"""You are linking a biomedical mention to the correct UMLS concept (CUI).
Pick exactly ONE candidate using the local context and the candidate metadata.
Return ONLY valid JSON with:
- selected_cui (string)
- selected_name (string)
- evidence_span (short exact quote from the context, or null)

MENTION: [{mention}]
LEFT_CONTEXT: {left_ctx}
RIGHT_CONTEXT: {right_ctx}
CANDIDATES: {json.dumps(candidates, ensure_ascii=False)}
"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": PICK_SCHEMA,
        "options": {"temperature": 0},
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    txt = r.json().get("response","").strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        return json.loads(m.group(0)) if m else {"selected_cui":"", "selected_name":"", "evidence_span": None}

if __name__ == "__main__":
    mention = "ASA"
    left = "Patient started on"
    right = "81 mg daily after NSTEMI."
    candidates = [
        {"cui":"C0004057","name":"Aspirin","types":["T109","T121"],"aliases":["ASA","acetylsalicylic acid"],"definition":"An analgesic and anti-inflammatory."},
        {"cui":"C4304040","name":"American Staffing Association","types":["ORG"],"aliases":["ASA"],"definition":"Trade association."}
    ]
    print(disambiguate(mention, left, right, candidates))
