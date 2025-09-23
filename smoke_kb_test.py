import re
import sys
import json
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

def create_sample_kb() -> dict:
    kb = {
        "aspirin": {
            "id": "UMLS:C0004057",
            "name": "Aspirin",
            "type": "Medication",
            "aliases": ["acetylsalicylic acid", "ASA"],
            "definition": "A salicylate used as an analgesic, antipyretic, anti-inflammatory, and antithrombotic agent."
        },
        "metoprolol": {
            "id": "UMLS:C0025859",
            "name": "Metoprolol",
            "type": "Medication",
            "aliases": ["metoprolol tartrate", "lopressor", "toprol"],
            "definition": "A selective beta-1 adrenergic receptor antagonist used for hypertension."
        },
        "lisinopril": {
            "id": "UMLS:C0065374",
            "name": "Lisinopril",
            "type": "Medication",
            "aliases": ["prinivil", "zestril"],
            "definition": "An ACE inhibitor used to treat hypertension and heart failure."
        },
        "morphine": {
            "id": "UMLS:C0026549",
            "name": "Morphine",
            "type": "Medication",
            "aliases": ["morphine sulfate"],
            "definition": "An opioid analgesic used for severe pain management."
        },

        "hypertension": {
            "id": "UMLS:C0020538",
            "name": "Hypertension",
            "type": "Disease",
            "aliases": ["high blood pressure", "HTN"],
            "definition": "Persistently elevated arterial blood pressure."
        },
        "pneumonia": {
            "id": "UMLS:C0032285",
            "name": "Pneumonia",
            "type": "Disease",
            "aliases": ["lung infection"],
            "definition": "Inflammation of the lung parenchyma."
        },
        "diabetes": {
            "id": "UMLS:C0011849",
            "name": "Diabetes Mellitus",
            "type": "Disease",
            "aliases": ["diabetes", "DM"],
            "definition": "A group of metabolic disorders characterized by hyperglycemia."
        },

        "chest pain": {
            "id": "UMLS:C0008031",
            "name": "Chest Pain",
            "type": "Symptom",
            "aliases": ["thoracic pain", "chest discomfort"],
            "definition": "Pain localized to the chest."
        },
        "shortness of breath": {
            "id": "UMLS:C0013404",
            "name": "Dyspnea",
            "type": "Symptom",
            "aliases": ["SOB", "breathlessness", "difficulty breathing"],
            "definition": "Difficult or labored breathing."
        },

        "echocardiogram": {
            "id": "UMLS:C0013516",
            "name": "Echocardiography",
            "type": "Procedure",
            "aliases": ["echo", "cardiac ultrasound"],
            "definition": "Ultrasound examination of the heart."
        },

        "azithromycin": {
            "id": "UMLS:AZI",
            "name": "Azithromycin",
            "type": "Medication",
            "aliases": ["zithromax"],
            "definition": "Macrolide antibiotic."
        },
        "digoxin": {
            "id": "UMLS:DIG",
            "name": "Digoxin",
            "type": "Medication",
            "aliases": [],
            "definition": "Cardiac glycoside."
        },
        "nevirapine": {
            "id": "UMLS:NEV",
            "name": "Nevirapine",
            "type": "Medication",
            "aliases": [],
            "definition": "Non-nucleoside reverse transcriptase inhibitor (NNRTI)."
        },
        "abacavir": {
            "id": "UMLS:ABA",
            "name": "Abacavir",
            "type": "Medication",
            "aliases": [],
            "definition": "Nucleoside reverse transcriptase inhibitor (NRTI)."
        },
        "zidovudine": {
            "id": "UMLS:ZDV",
            "name": "Zidovudine",
            "type": "Medication",
            "aliases": ["AZT"],
            "definition": "Nucleoside reverse transcriptase inhibitor (NRTI)."
        },
        "ampicillin": {
            "id": "UMLS:AMP",
            "name": "Ampicillin",
            "type": "Medication",
            "aliases": [],
            "definition": "Penicillin-class beta-lactam antibiotic."
        },
        "acetaminophen": {
            "id": "UMLS:APAP",
            "name": "Acetaminophen",
            "type": "Medication",
            "aliases": ["tylenol", "paracetamol", "APAP"],
            "definition": "Analgesic and antipyretic."
        },
        "clonazepam": {
            "id": "UMLS:CLZ",
            "name": "Clonazepam",
            "type": "Medication",
            "aliases": ["klonopin"],
            "definition": "Benzodiazepine."
        },
        "escitalopram": {
            "id": "UMLS:ESC",
            "name": "Escitalopram",
            "type": "Medication",
            "aliases": ["lexapro"],
            "definition": "Selective serotonin reuptake inhibitor (SSRI)."
        },
        "nortriptyline": {
            "id": "UMLS:NOR",
            "name": "Nortriptyline",
            "type": "Medication",
            "aliases": [],
            "definition": "Tricyclic antidepressant (TCA)."
        },
        "rosuvastatin": {
            "id": "UMLS:RSV",
            "name": "Rosuvastatin",
            "type": "Medication",
            "aliases": ["crestor"],
            "definition": "HMG-CoA reductase inhibitor (statin)."
        },
        "quetiapine": {
            "id": "UMLS:QTP",
            "name": "Quetiapine",
            "type": "Medication",
            "aliases": ["seroquel"],
            "definition": "Atypical antipsychotic."
        },
        "trazodone": {
            "id": "UMLS:TRZ",
            "name": "Trazodone",
            "type": "Medication",
            "aliases": [],
            "definition": "Serotonin antagonist and reuptake inhibitor."
        },
        "zolpidem": {
            "id": "UMLS:ZLP",
            "name": "Zolpidem",
            "type": "Medication",
            "aliases": ["ambien"],
            "definition": "Non-benzodiazepine hypnotic."
        },
        "insulin": {
            "id": "UMLS:INS",
            "name": "Insulin",
            "type": "Medication",
            "aliases": [],
            "definition": "Peptide hormone; antidiabetic agent."
        },
        "nicardipine": {
            "id": "UMLS:NIC",
            "name": "Nicardipine",
            "type": "Medication",
            "aliases": [],
            "definition": "Dihydropyridine calcium channel blocker."
        },
        "nitroprusside": {
            "id": "UMLS:NTP",
            "name": "Nitroprusside",
            "type": "Medication",
            "aliases": ["sodium nitroprusside"],
            "definition": "Potent arterial and venous vasodilator."
        },
        "dexamethasone": {
            "id": "UMLS:DEX",
            "name": "Dexamethasone",
            "type": "Medication",
            "aliases": ["decadron"],
            "definition": "Glucocorticoid corticosteroid."
        },

        "coronary artery disease": {
            "id": "UMLS:CAD",
            "name": "Coronary Artery Disease",
            "type": "Disease",
            "aliases": ["CAD", "atherosclerotic heart disease"],
            "definition": "Atherosclerotic narrowing of coronary arteries."
        },
        "hyperlipidemia": {
            "id": "UMLS:HLD",
            "name": "Hyperlipidemia",
            "type": "Disease",
            "aliases": ["HLD", "dyslipidemia"],
            "definition": "Elevated blood lipids."
        },
        "aortic stenosis": {
            "id": "UMLS:AS",
            "name": "Aortic Stenosis",
            "type": "Disease",
            "aliases": ["AS"],
            "definition": "Narrowing of the aortic valve."
        },
        "meningioma": {
            "id": "UMLS:MNG",
            "name": "Meningioma",
            "type": "Disease",
            "aliases": [],
            "definition": "Generally benign tumor of the meninges."
        },
        "seizure": {
            "id": "UMLS:SEZ",
            "name": "Seizure",
            "type": "Symptom",
            "aliases": ["convulsion"],
            "definition": "Paroxysmal neurologic event due to abnormal neuronal activity."
        },
        "hypothyroidism": {
            "id": "UMLS:HYPOTH",
            "name": "Hypothyroidism",
            "type": "Disease",
            "aliases": ["hypothyroid"],
            "definition": "Deficiency of thyroid hormones."
        },

        "cabg": {
            "id": "UMLS:CABG",
            "name": "Coronary Artery Bypass Grafting",
            "type": "Procedure",
            "aliases": ["CABG"],
            "definition": "Surgical revascularization using grafts."
        },
        "craniotomy": {
            "id": "UMLS:CRAN",
            "name": "Craniotomy",
            "type": "Procedure",
            "aliases": [],
            "definition": "Surgical opening of the skull."
        },
        "angiography": {
            "id": "UMLS:ANG",
            "name": "Angiography",
            "type": "Procedure",
            "aliases": ["balloon angiography"],
            "definition": "Imaging of blood vessels using contrast."
        },
        "ct": {
            "id": "UMLS:CT",
            "name": "Computed Tomography",
            "type": "Test",
            "aliases": ["ct scan", "head ct"],
            "definition": "X-ray based cross-sectional imaging."
        },
        "mri": {
            "id": "UMLS:MRI",
            "name": "Magnetic Resonance Imaging",
            "type": "Test",
            "aliases": ["mri", "mri scan"],
            "definition": "Magnetic field and radiofrequency based imaging."
        },

    }
    return kb

DEID = re.compile(r'\[\*\*.*?\*\*\]')
ABBREV_TYPES = {
    "CAD": "Disease", "HTN": "Disease", "HLD": "Disease", "AS": "Disease",
    "CABG": "Procedure", "CT": "Test", "MRI": "Test",
}
MED_SIGNAL = re.compile(
    r'\b([A-Z][a-z][a-zA-Z\-]+)'            
    r'(?:\s+\d+\s?(?:mg|mcg|g))?'          
    r'(?:\s*(?:PO|IV|IM|SC|qHS|BID|TID|QID|PRN))?\b'
)

def normalize_text(s: str) -> str:
    s = DEID.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fuzzy_ratio(a, b):
    return int(100 * SequenceMatcher(None, a.lower(), b.lower()).ratio())

def build_med_index(kb: dict):
    meds = set()
    for v in kb.values():
        if v.get("type") == "Medication":
            meds.add(v["name"].lower())
            for a in v.get("aliases", []):
                meds.add(a.lower())
    return meds

def extract_entities_fallback(text: str, med_index: set) -> List[Dict]:
    text = normalize_text(text)
    entities = []

    for abbr, typ in ABBREV_TYPES.items():
        for m in re.finditer(rf'\b{re.escape(abbr)}\b', text):
            entities.append({"text": abbr, "type": typ, "start": m.start(), "end": m.end()})

    for kw in ["pneumonia", "meningioma", "seizure", "hypothyroid", "hypertension"]:
        for m in re.finditer(rf'\b{kw}\b', text, flags=re.I):
            entities.append({"text": m.group(0), "type": "Disease", "start": m.start(), "end": m.end()})

    skip_words = {
        "patient","head","right","left","pain","daily","dose","plan","goal","meds","surgery",
        "reports","with","then","the","will","hpi","bp","mac","control","started","include",
        "mass"
    }
    for m in MED_SIGNAL.finditer(text):
        token = m.group(1)
        low = token.lower()
        if low in skip_words:
            continue

        # Look ahead for a nearby dose unit to qualify non-KB tokens
        span_end = m.end()
        window = text[span_end: span_end + 12]  # small window to catch " 10 mg"
        has_dose = bool(re.search(r'\b\d+\s?(?:mg|mcg|g)\b', window, flags=re.I))

        if (low in med_index) or has_dose:
            entities.append({"text": token, "type": "Medication", "start": m.start(1), "end": m.end(1)})

    # De-dup by span+type
    seen = set()
    uniq = []
    for e in entities:
        key = (e["start"], e["end"], e["type"])
        if key not in seen and e["start"] != -1:
            uniq.append(e); seen.add(key)
    return uniq

def find_candidates(entity_text: str, entity_type: str, kb: dict, threshold: int = 65, top_k: int = 5) -> List[Dict]:
    q = entity_text.strip().lower()
    cands = []
    for _, entry in kb.items():
        names = [entry['name']] + entry.get('aliases', [])
        best = max((fuzzy_ratio(q, n) for n in names), default=0)
        score = best + (10 if entity_type == entry['type'] else 0)
        if score >= threshold:
            cands.append({**entry, "score": score})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:top_k]

SAMPLE_TEXTS = [
    "Diabetic patient with pneumonia. Pain managed with morphine. Patient has history of hypertension, currently on lisinopril.",
    "HPI: 75yo with CAD s/p CABG x4, AS, HLD. MRI showed frontal mass; started on Decadron. Meds include Klonopin, Lexapro, Nortriptyline, Crestor, Seroquel, Trazodone, Ambien. Nicardipine gtt for BP control.",
    "The patient will not be discharged on MAC prophylaxis (Azithromycin) and Digoxin. Will continue antibiotics for pneumonia."
]

def run_smoke() -> Tuple[int, dict]:
    kb = create_sample_kb()
    med_index = build_med_index(kb)
    total, linked = 0, 0
    detailed = []

    for i, text in enumerate(SAMPLE_TEXTS, 1):
        ents = extract_entities_fallback(text, med_index)
        results = []
        for e in ents:
            total += 1
            cands = find_candidates(e["text"], e["type"], kb)
            best = cands[0] if cands else None
            if best:
                linked += 1
            results.append({
                "mention": e["text"],
                "type": e["type"],
                "linked": bool(best),
                "best_name": best["name"] if best else None,
                "best_id": best["id"] if best else None,
                "score": best["score"] if best else None
            })
        detailed.append({"text_idx": i, "text": text, "entities": results})

    summary = {
        "total_entities": total,
        "linked_entities": linked,
        "link_rate": round(linked / total, 3) if total else 0.0,
        "cases": detailed
    }
    return (0 if linked > 0 else 1), summary


def main():
    exit_code, summary = run_smoke()
    print("=== SMOKE TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if exit_code == 0:
        print("\nOK: KB linking works on sample texts.")
    else:
        print("\nFAIL: No entities were linked. Check KB coverage or extract patterns.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
