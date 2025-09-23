import re
import json
import getpass
import warnings
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import ollama
import psycopg2
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')


DEID = re.compile(r'\[\*\*.*?\*\*\]')
WS = re.compile(r'\s+')

ABBREV_TYPES = {
    "CAD": "Disease", "HTN": "Disease", "HLD": "Disease", "DM": "Disease", "AS": "Disease", "CHF": "Disease",
    "CABG": "Procedure", "PCI": "Procedure", "CT": "Test", "MRI": "Test", "CXR": "Test", "EKG": "Test", "ECG": "Test",
}

MED_SIGNAL = re.compile(
    r'\b([A-Z][a-z][a-zA-Z\-]+)'           
    r'(?:\s+\d+\s?(?:mg|mcg|g|units|mEq|IU))?'  
    r'(?:\s*(?:PO|IV|IM|SC|SL|PR|qHS|BID|TID|QID|PRN))?\b'
)

DOSE_NEARBY = re.compile(r'\b\d+\s?(?:mg|mcg|g|units|mEq|IU)\b', flags=re.I)

def normalize_text(s: str) -> str:
    s = DEID.sub('', s)
    s = WS.sub(' ', s).strip()
    return s

def fuzzy_ratio(a: str, b: str) -> int:
    return int(100 * SequenceMatcher(None, a.lower(), b.lower()).ratio())

class DynamicKnowledgeBase:

    def __init__(self, engine):
        self.engine = engine
        self.kb: Dict[str, Dict] = {}
        self.med_index = set()
        self.diag_index = set()
        self.proc_index = set()
        self.test_index = set()

    def _add_entry(self, key: str, entry: Dict):
        k = key.strip().lower()
        if not k:
            return
        if k not in self.kb:
            self.kb[k] = entry

    @staticmethod
    def _alias_variants(name: str) -> List[str]:
        aliases = set()
        n = name.strip()
        aliases.add(n)
        aliases.add(n.lower())
        aliases.add(n.upper())
        aliases.add(n.title())
        aliases.add(re.sub(r'[^\w\s]', ' ', n).strip())
        aliases.add(re.sub(r'[^\w\s]', '', n).strip())

        aliases.add(WS.sub(' ', n))
        return [a for a in {x for x in aliases if x}]

    def load_medications(self, max_rows: int = 500000):

        q = f"""
        SELECT DISTINCT
            COALESCE(LOWER(drug_name_generic), LOWER(drug)) AS gname,
            LOWER(drug) AS brand
        FROM prescriptions
        WHERE COALESCE(drug_name_generic, drug) IS NOT NULL
        LIMIT {max_rows}
        """
        df = pd.read_sql(q, self.engine)
        for _, r in df.iterrows():
            g = (r['gname'] or '').strip()
            b = (r['brand'] or '').strip()
            if not g and not b:
                continue
            canon = g or b
            aliases = set(self._alias_variants(canon))
            if b and b != canon:
                aliases.update(self._alias_variants(b))
            entry = {
                'id': f"MED:{canon}",
                'name': canon,
                'type': 'Medication',
                'aliases': sorted({a for a in aliases if a}),
                'definition': None
            }
            self._add_entry(canon, entry)
            self.med_index.add(canon.lower())
            if b and b != canon:
                self._add_entry(b, entry)

    def load_diagnoses(self, max_rows: int = 500000):

        q = f"""
        SELECT DISTINCT d.icd9_code, LOWER(d.long_title) AS title
        FROM d_icd_diagnoses d
        WHERE d.long_title IS NOT NULL
        LIMIT {max_rows}
        """
        df = pd.read_sql(q, self.engine)
        for _, r in df.iterrows():
            code = (r['icd9_code'] or '').strip()
            title = (r['title'] or '').strip()
            if not title:
                continue
            aliases = set(self._alias_variants(title))
            entry = {
                'id': f"ICD9:{code}",
                'name': title,
                'type': 'Disease',
                'aliases': sorted({a for a in aliases if a}),
                'definition': None
            }
            self._add_entry(title, entry)
            self.diag_index.add(title.lower())

    def load_procedures(self, max_rows: int = 500000):

        q = f"""
        SELECT DISTINCT d.icd9_code, LOWER(d.long_title) AS title
        FROM d_icd_procedures d
        WHERE d.long_title IS NOT NULL
        LIMIT {max_rows}
        """
        df = pd.read_sql(q, self.engine)
        for _, r in df.iterrows():
            code = (r['icd9_code'] or '').strip()
            title = (r['title'] or '').strip()
            if not title:
                continue
            aliases = set(self._alias_variants(title))
            entry = {
                'id': f"ICD9PROC:{code}",
                'name': title,
                'type': 'Procedure',
                'aliases': sorted({a for a in aliases if a}),
                'definition': None
            }
            self._add_entry(title, entry)
            self.proc_index.add(title.lower())

    def load_tests(self, max_rows: int = 500000):

        q = f"""
        SELECT DISTINCT LOWER(label) AS label
        FROM d_labitems
        WHERE label IS NOT NULL
        LIMIT {max_rows}
        """
        try:
            df = pd.read_sql(q, self.engine)
            for _, r in df.iterrows():
                lab = (r['label'] or '').strip()
                if not lab:
                    continue
                aliases = set(self._alias_variants(lab))
                entry = {
                    'id': f"LAB:{lab}",
                    'name': lab,
                    'type': 'Test',
                    'aliases': sorted({a for a in aliases if a}),
                    'definition': None
                }
                self._add_entry(lab, entry)
                self.test_index.add(lab.lower())
        except Exception:
            pass

    def build(self):
        print("→ Building dynamic KB from MIMIC tables…")
        self.load_medications()
        self.load_diagnoses()
        self.load_procedures()
        self.load_tests()
        print(f"   KB entries: {len(self.kb)}")

class BiomedicalEntityLinker:
    def __init__(self, engine, model_name="gemma3:1b"):
        self.model_name = model_name
        self.setup_model()

        self.entity_types = ["Medication", "Disease", "Symptom", "Procedure", "Anatomy", "Test", "Other"]

        self.kb = DynamicKnowledgeBase(engine)
        self.kb.build()

    def setup_model(self):
        try:
            response = ollama.list()
            print("✓ Ollama is running")

            model_names = []
            if 'models' in response:
                model_names = [model.get('name', '') for model in response['models']]

            if self.model_name not in model_names:
                try:
                    ollama.pull(self.model_name)
                    print(f"✓ Successfully downloaded {self.model_name}")
                except Exception:
                    if model_names and model_names[0]:
                        self.model_name = model_names[0]
                        print(f"Using available model: {self.model_name}")

            print(f"✓ Model {self.model_name} ready")
        except Exception as e:
            print(f"Ollama setup failed: {e}")
            raise RuntimeError("Ollama not available")

    def generate_response(self, prompt: str, max_tokens: int = 400) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            )
            return response.get('response', '').strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def extract_entities(self, clinical_text: str, max_length: int = 2000) -> List[Dict]:
        text = normalize_text(clinical_text[:max_length])

        entity_types_str = ", ".join(self.entity_types)
        prompt = f"""Extract biomedical entities from the clinical text.

Return STRICT JSON with this schema:
{{
  "entities": [{{"text": str, "type": str}}]
}}
- Types must be one of: {entity_types_str}
- Do NOT include any keys other than exactly "entities".

Clinical text:
\"\"\"{text}\"\"\""""

        entities: List[Dict] = []

        raw = self.generate_response(prompt, max_tokens=500)
        try:
            data = json.loads(raw)
            for item in data.get("entities", []):
                etxt = (item.get("text") or "").strip()
                etype = (item.get("type") or "").strip()
                if etxt and etype in self.entity_types:
                    start = text.lower().find(etxt.lower())
                    end = start + len(etxt) if start >= 0 else -1
                    entities.append({"text": etxt, "type": etype, "start": start, "end": end})
        except Exception:
            pass
        if not entities:
            entities = self._extract_entities_fallback(text)

        seen, uniq = set(), []
        for e in entities:
            if e["start"] == -1:
                continue
            key = (e["start"], e["end"], e["type"])
            if key not in seen:
                uniq.append(e)
                seen.add(key)

        return uniq

    def _extract_entities_fallback(self, text: str) -> List[Dict]:
        ents: List[Dict] = []

        for abbr, typ in ABBREV_TYPES.items():
            for m in re.finditer(rf'\b{re.escape(abbr)}\b', text):
                ents.append({"text": abbr, "type": typ, "start": m.start(), "end": m.end()})

        for kw in ["pneumonia", "sepsis", "meningioma", "stroke", "mi", "myocardial infarction",
                   "diabetes", "hypertension", "hyperlipidemia", "hypothyroid", "heart failure"]:
            for m in re.finditer(rf'\b{kw}\b', text, flags=re.I):
                ents.append({"text": m.group(0), "type": "Disease", "start": m.start(), "end": m.end()})

        for kw in ["echocardiogram", "echo", "ct", "mri", "cxr", "x-ray", "ekg", "ecg"]:
            for m in re.finditer(rf'\b{kw}\b', text, flags=re.I):
                ents.append({"text": m.group(0), "type": "Test", "start": m.start(), "end": m.end()})

        for kw in ["cabg", "angiography", "pci", "craniotomy", "thoracotomy", "intubation"]:
            for m in re.finditer(rf'\b{kw}\b', text, flags=re.I):
                ents.append({"text": m.group(0), "type": "Procedure", "start": m.start(), "end": m.end()})

        skip_words = {
            "patient", "right", "left", "daily", "dose", "plan", "goal", "meds", "surgery", "reports",
            "with", "then", "the", "will", "hpi", "bp", "mac", "control", "started", "include", "mass"
        }
        for m in MED_SIGNAL.finditer(text):
            token = m.group(1)
            low = token.lower()
            if low in skip_words:
                continue
            nearby = text[m.end(): m.end() + 16]
            has_dose = bool(DOSE_NEARBY.search(nearby))
            in_kb = (low in self.kb.med_index)
            if in_kb or has_dose:
                ents.append({"text": token, "type": "Medication", "start": m.start(1), "end": m.end(1)})

        return ents


    def _find_candidates(self, mention: str, etype: Optional[str], top_k: int = 8) -> List[Dict]:

        q = mention.strip().lower()
        cands: List[Dict] = []
        for entry in self.kb.kb.values():
            names = [entry['name']] + entry.get('aliases', [])
            best = max((fuzzy_ratio(q, n) for n in names), default=0)
            score = best + (10 if etype and etype == entry['type'] else 0)
            if score >= 70 or (etype in ("Medication", "Test", "Procedure") and score >= 65):
                cands.append({**entry, "score": score})
        cands.sort(key=lambda x: x["score"], reverse=True)
        return cands[:top_k]

    def _disambiguate_with_llm(self, entity_text: str, context: str, candidates: List[Dict]) -> Optional[Dict]:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        buf = []
        for i, c in enumerate(candidates[:6], 1):
            line = f"{i}. {c['name']} ({c['id']}) - {c['type']}"
            buf.append(line)
        cand_txt = "\n".join(buf)

        prompt = f"""Pick the best candidate for the mention "{entity_text}" in this context:

Context:
\"\"\"{context[:500]}\"\"\"

Candidates:
{cand_txt}

Answer with ONLY the number (e.g., "2")."""
        resp = self.generate_response(prompt, max_tokens=20)
        try:
            choice = int(re.search(r'\d+', resp).group())
            if 1 <= choice <= len(candidates[:6]):
                return candidates[choice - 1]
        except Exception:
            pass
        return candidates[0]

    def link_entities(self, clinical_text: str) -> List[Dict]:
        ents = self.extract_entities(clinical_text)
        linked = []
        for e in ents:
            cands = self._find_candidates(e['text'], e['type'])
            best = self._disambiguate_with_llm(
                e['text'],
                clinical_text[max(0, e['start'] - 120): e['end'] + 120],
                cands
            )
            linked.append({
                'mention': e['text'],
                'type': e['type'],
                'start': e['start'],
                'end': e['end'],
                'linked_entity': best,
                'candidates': cands
            })
        return linked

class MIMICProcessor:
    def __init__(self, db_config: Dict = None):
        self.db_config = db_config
        self.engine = None
        if db_config:
            self.setup_connection()

    def setup_connection(self):
        try:
            print("Attempting to connect to PostgreSQL database...")
            conn_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@" \
                          f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.engine = create_engine(
                conn_string,
                connect_args={"connect_timeout": 10},
                pool_timeout=20
            )
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
            print("✓ Successfully connected to MIMIC-III PostgreSQL database")
        except Exception as e:
            print(f"✗ Failed to connect to PostgreSQL: {e}")
            self.engine = None

    @staticmethod
    def get_db_config() -> Dict:
        print("\n=== MIMIC-III PostgreSQL Configuration ===")
        return {
            'host': 'localhost',
            'port': '5432',
            'database': 'mimic',
            'user': input("PostgreSQL username: ").strip() or 'postgres',
            'password': getpass.getpass("PostgreSQL password: ")
        }

    def load_mimic_notes(self, limit: int = 50) -> pd.DataFrame:
        if not self.engine:
            raise RuntimeError("No database connection available.")
        q = f"""
        SELECT subject_id, hadm_id, category, text
        FROM noteevents
        WHERE category IN ('Discharge summary', 'Physician', 'Nursing')
          AND (iserror IS DISTINCT FROM '1' OR iserror IS NULL)
          AND text IS NOT NULL
          AND LENGTH(text) BETWEEN 200 AND 8000
        ORDER BY subject_id
        LIMIT {limit}
        """
        print(f"Loading {limit} clinical notes for entity linking...")
        df = pd.read_sql_query(q, self.engine)
        df['text'] = df['text'].astype(str).map(normalize_text)
        print(f"✓ Loaded {len(df)} notes")
        return df

def evaluate_entity_linking(predicted_links: List[Dict]) -> Dict:
    successful_links = sum(1 for link in predicted_links if link['linked_entity'] is not None)
    total_entities = len(predicted_links)
    return {
        'total_entities': total_entities,
        'successful_links': successful_links,
        'linking_accuracy': successful_links / total_entities if total_entities > 0 else 0.0
    }

def main():
    print("=== MIMIC-III Biomedical Entity Linking (Dynamic KB from Postgres) ===\n")

    # DB setup
    use_real_data = input("Connect to MIMIC-III PostgreSQL database? (y/n): ").lower().startswith('y')
    if not use_real_data:
        print("This build requires Postgres to construct the KB. Exiting.")
        return None

    try:
        db_config = MIMICProcessor.get_db_config()
        processor = MIMICProcessor(db_config)
        if not processor.engine:
            print("Could not connect; exiting.")
            return None
    except Exception as e:
        print(f"Database setup failed: {e}")
        return None

    default_limit = 8
    try:
        limit = int(input(f"\nNumber of notes to process [{default_limit}]: ") or default_limit)
    except ValueError:
        limit = default_limit

    print("\nInitializing Biomedical Entity Linker (dynamic KB)…")
    try:
        linker = BiomedicalEntityLinker(engine=processor.engine)
    except RuntimeError as e:
        print(f"Failed to initialize entity linker: {e}")
        return None

    print(f"\nLoading clinical notes…")
    notes_df = processor.load_mimic_notes(limit=limit)
    print(f"Loaded {len(notes_df)} notes\n")

    results = []
    print("Starting biomedical entity linking…\n")
    for idx, row in notes_df.iterrows():
        subject_id = row['subject_id']
        text = row['text']
        category = row.get('category', 'Unknown')

        print(f"Processing Subject {subject_id} - {category}")
        print(f"Text preview: {text[:220]}...")

        linked_entities = linker.link_entities(text)

        print(f"Found {len(linked_entities)} entities:")
        for entity in linked_entities[:25]:
            linked_name = entity['linked_entity']['name'] if entity['linked_entity'] else 'UNLINKED'
            linked_id = entity['linked_entity']['id'] if entity['linked_entity'] else 'N/A'
            print(f"  • {entity['mention']} ({entity['type']}) → {linked_name} ({linked_id})")

        result = {
            'subject_id': int(subject_id),
            'category': category,
            'text': text,
            'entities': linked_entities,
            'metrics': evaluate_entity_linking(linked_entities)
        }
        results.append(result)
        print(f"  Linking accuracy (this note): {result['metrics']['linking_accuracy']:.2f}\n")

    print("=== OVERALL RESULTS ===")
    total_entities = sum(r['metrics']['total_entities'] for r in results)
    successful_links = sum(r['metrics']['successful_links'] for r in results)
    overall_accuracy = successful_links / total_entities if total_entities > 0 else 0.0
    print(f"Total entities found: {total_entities}")
    print(f"Successfully linked: {successful_links}")
    print(f"Overall linking accuracy: {overall_accuracy:.3f}")

    entity_types = defaultdict(int)
    linked_types = defaultdict(int)
    for result in results:
        for entity in result['entities']:
            entity_types[entity['type']] += 1
            if entity['linked_entity']:
                linked_types[entity['type']] += 1

    print("\nEntity Type Breakdown:")
    for etype, count in entity_types.items():
        linked_count = linked_types[etype]
        acc = linked_count / count if count > 0 else 0
        print(f"  {etype}: {linked_count}/{count} ({acc:.2f})")

    with open("gemma_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✓ Results saved to: gemma3_results.json")
    return results


if __name__ == "__main__":
    main()
