import re
import json
import getpass
import warnings
from typing import List, Dict, Optional
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd
import ollama
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

DEID = re.compile(r'\[\*\*.*?\*\*\]')
WS = re.compile(r'\s+')
ABBREV_TYPES = {
    "CAD": "Disease", "HTN": "Disease", "HLD": "Disease", "DM": "Disease", "AS": "Disease", "CHF": "Disease",
    "CABG": "Procedure", "PCI": "Procedure", "CT": "Test", "MRI": "Test", "CXR": "Test", "EKG": "Test", "ECG": "Test",
}
MED_SIGNAL = re.compile(r'\b([A-Z][a-z][a-zA-Z\-]+)(?:\s+\d+\s?(?:mg|mcg|g|units|mEq|IU))?(?:\s*(?:PO|IV|IM|SC|SL|PR|qHS|BID|TID|QID|PRN))?\b')
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
        self.alias_to_entry: Dict[str, Dict] = {}
        self.prefix_index: Dict[str, List[Dict]] = defaultdict(list)

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
        if not n:
            return []
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
                'definition': None,
                'source': 'prescriptions'
            }
            self._add_entry(canon, entry)
            self.med_index.add(canon.lower())
            if b and b != canon:
                self._add_entry(b, entry)

    def load_med_admin(self, max_rows: int = 500000):
        queries = [
            f"""
            SELECT DISTINCT LOWER(di.label) AS name
            FROM inputevents_mv iv
            JOIN d_items di ON iv.itemid = di.itemid
            WHERE di.label IS NOT NULL
            LIMIT {max_rows}
            """,
            f"""
            SELECT DISTINCT LOWER(di.label) AS name
            FROM inputevents_cv ic
            JOIN d_items di ON ic.itemid = di.itemid
            WHERE di.label IS NOT NULL
            LIMIT {max_rows}
            """
        ]
        for q in queries:
            try:
                df = pd.read_sql(q, self.engine)
                for _, r in df.iterrows():
                    name = (r['name'] or '').strip()
                    if not name:
                        continue
                    aliases = set(self._alias_variants(name))
                    entry = {
                        'id': f"ADMINMED:{name}",
                        'name': name,
                        'type': 'Medication',
                        'aliases': sorted(aliases),
                        'definition': None,
                        'source': 'inputevents'
                    }
                    self._add_entry(name, entry)
                    self.med_index.add(name.lower())
            except Exception:
                pass

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
                'definition': None,
                'source': 'diagnosis'
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
                'definition': None,
                'source': 'procedure'
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
                    'definition': None,
                    'source': 'dlabitems'
                }
                self._add_entry(lab, entry)
                self.test_index.add(lab.lower())
        except Exception:
            pass

    def load_lab_hints_from_events(self, max_rows: int = 500000):
        try:
            q = f"""
            SELECT DISTINCT LOWER(di.label) AS name
            FROM labevents le
            JOIN d_labitems di ON le.itemid = di.itemid
            WHERE di.label IS NOT NULL
            LIMIT {max_rows}
            """
            df = pd.read_sql(q, self.engine)
            for _, r in df.iterrows():
                name = (r['name'] or '').strip()
                if not name:
                    continue
                aliases = set(self._alias_variants(name))
                entry = {
                    'id': f"LABEV:{name}",
                    'name': name,
                    'type': 'Test',
                    'aliases': sorted(aliases),
                    'definition': None,
                    'source': 'labevents'
                }
                self._add_entry(name, entry)
                self.test_index.add(name.lower())
        except Exception:
            pass

    def load_microbiology(self, max_rows: int = 500000):
        try:
            q = f"""
            SELECT DISTINCT LOWER(organism_name) AS organism, LOWER(antibiotic_name) AS abx
            FROM microbiologyevents
            WHERE organism_name IS NOT NULL OR antibiotic_name IS NOT NULL
            LIMIT {max_rows}
            """
            df = pd.read_sql(q, self.engine)
            for _, r in df.iterrows():
                for field in ('organism', 'abx'):
                    val = (r[field] or '').strip()
                    if not val:
                        continue
                    aliases = set(self._alias_variants(val))
                    entry = {
                        'id': f"MICRO:{val}",
                        'name': val,
                        'type': 'Test',
                        'aliases': sorted(aliases),
                        'definition': None,
                        'source': 'microbiology'
                    }
                    self._add_entry(val, entry)
                    self.test_index.add(val.lower())
        except Exception:
            pass

    def build(self):
        self.load_medications()
        self.load_med_admin()
        self.load_diagnoses()
        self.load_procedures()
        self.load_tests()
        self.load_lab_hints_from_events()
        self.load_microbiology()
        for entry in self.kb.values():
            names = [entry['name']] + entry.get('aliases', [])
            for n in names:
                nlow = (n or "").strip().lower()
                if not nlow:
                    continue
                self.alias_to_entry[nlow] = entry
                pref = nlow[:2]
                if pref:
                    self.prefix_index[pref].append(entry)

class BiomedicalEntityLinker:
    SOURCE_WEIGHTS = {
        'inputevents': 8,
        'prescriptions': 5,
        'diagnosis': 6,
        'procedure': 6,
        'dlabitems': 4,
        'labevents': 3,
        'microbiology': 2,
        None: 0
    }

    def __init__(self, engine, model_name="gemma3:1b"):
        self.model_name = model_name
        self.setup_model()
        self.entity_types = ["Medication", "Disease", "Symptom", "Procedure", "Anatomy", "Test", "Other"]
        self.kb = DynamicKnowledgeBase(engine)
        self.kb.build()

    def setup_model(self):
        try:
            resp = ollama.list()
            names = [m.get('name', '') for m in resp.get('models', [])] if isinstance(resp, dict) else []
            if self.model_name not in names:
                try:
                    ollama.pull(self.model_name)
                except Exception:
                    if names and names[0]:
                        self.model_name = names[0]
        except Exception as e:
            raise RuntimeError("Ollama not available")

    def generate_response(self, prompt: str, max_tokens: int = 300) -> str:
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
        except Exception:
            return ""

    def _extract_llm_json(self, text: str, entity_types_str: str) -> List[Dict]:
        prompt = f"""Extract biomedical entities from the clinical text.

        Return STRICT JSON exactly as:
        {{"entities":[{{"text":"...", "type":"..."}}]}}

        Rules:
        - Types must be one of: {entity_types_str}
        - Return at most 40 entities for this chunk.
        - Deduplicate case-insensitively.
        - Ignore PHI markers, dates, generic words, and section headers.

        Text:
        \"\"\"{text}\"\"\""""

        raw = ""
        try:
            resp = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format='json',
                options={'num_predict': 512, 'temperature': 0.1, 'top_p': 0.9}
            )
            raw = resp.get('response', '').strip()
        except Exception:
            return []

        try:
            m = re.search(r'\{.*\}\s*$', raw, flags=re.S)
            if m:
                raw = m.group(0)
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception:
            return []

    def extract_entities(self, clinical_text: str, max_length: int = 8000) -> List[Dict]:
        text = normalize_text(clinical_text[:max_length])
        entity_types_str = ", ".join(self.entity_types)
        CHUNK_SIZE = 1200
        OVERLAP = 150
        MAX_ENTS = 60
        i = 0
        all_ents: List[Dict] = []
        while i < len(text) and len(all_ents) < MAX_ENTS:
            piece = text[i:i+CHUNK_SIZE]
            items = self._extract_llm_json(piece, entity_types_str)
            for item in items:
                etxt = (item.get("text") or "").strip()
                etype = (item.get("type") or "").strip()
                if not etxt or etype not in self.entity_types:
                    continue
                local = piece.lower().find(etxt.lower())
                if local < 0:
                    continue
                start = i + local
                end = start + len(etxt)
                all_ents.append({"text": etxt, "type": etype, "start": start, "end": end})
            i += CHUNK_SIZE - OVERLAP
            if len(all_ents) >= MAX_ENTS:
                break

        seen, uniq = set(), []
        for e in all_ents:
            if e["start"] == -1:
                continue
            key = (e["start"], e["end"], e["type"])
            if key not in seen:
                uniq.append(e)
                seen.add(key)
        return uniq[:MAX_ENTS]

    def _find_candidates(self, mention: str, etype: Optional[str], top_k: int = 8) -> List[Dict]:
        q = (mention or "").strip().lower()
        if not q:
            return []
        hit = self.kb.alias_to_entry.get(q)
        if hit:
            score = 100 + (10 if etype and etype == hit['type'] else 0) + self.SOURCE_WEIGHTS.get(hit.get('source'), 0)
            return [{**hit, "score": score}]
        bucket = self.kb.prefix_index.get(q[:2], []) or self.kb.prefix_index.get(q[:1], [])
        cands: List[Dict] = []
        for entry in bucket[:500]:
            names = [entry['name']] + entry.get('aliases', [])
            best = max((fuzzy_ratio(q, n) for n in names), default=0)
            score = best + (10 if etype and etype == entry['type'] else 0) + self.SOURCE_WEIGHTS.get(entry.get('source'), 0)
            if score >= 75 or (etype in ("Medication", "Test", "Procedure") and score >= 70):
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
            buf.append(f"{i}. {c['name']} ({c['id']}) - {c['type']}")
        cand_txt = "\n".join(buf)
        prompt = f"""Pick the best candidate for the mention "{entity_text}" in this context:

Context:
\"\"\"{context[:500]}\"\"\"

Candidates:
{cand_txt}

Answer with ONLY the number."""
        resp = self.generate_response(prompt, max_tokens=16)
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
            conn_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.engine = create_engine(conn_string, connect_args={"connect_timeout": 10}, pool_timeout=20)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).fetchone()
        except Exception:
            self.engine = None

    @staticmethod
    def get_db_config() -> Dict:
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
        df = pd.read_sql_query(q, self.engine)
        df['text'] = df['text'].astype(str).map(normalize_text)
        return df

def evaluate_entity_linking(predicted_links: List[Dict]) -> Dict:
    successful_links = sum(1 for link in predicted_links if link['linked_entity'] is not None)
    total_entities = len(predicted_links)
    return {
        'total_entities': total_entities,
        'successful_links': successful_links,
        'linking_accuracy': successful_links / total_entities if total_entities > 0 else 0.0
    }

def summarize(results: List[Dict]) -> Dict:
    total_entities = sum(r['metrics']['total_entities'] for r in results)
    successful = sum(r['metrics']['successful_links'] for r in results)
    overall_acc = successful / total_entities if total_entities else 0.0
    entity_types = defaultdict(int)
    linked_types = defaultdict(int)
    for r in results:
        for e in r['entities']:
            entity_types[e['type']] += 1
            if e['linked_entity']:
                linked_types[e['type']] += 1
    by_type = []
    for et, count in entity_types.items():
        linked = linked_types.get(et, 0)
        acc = linked / count if count else 0.0
        by_type.append({"type": et, "total": count, "linked": linked, "accuracy": acc})
    notes_brief = []
    for r in results:
        m = r["metrics"]
        acc = (m["successful_links"] / m["total_entities"]) if m["total_entities"] else 0.0
        notes_brief.append({"subject_id": r["subject_id"], "category": r["category"], "entities": m["total_entities"], "linked": m["successful_links"], "accuracy": acc})
    return {
        "notes_processed": len(results),
        "total_entities": total_entities,
        "successful_links": successful,
        "overall_accuracy": overall_acc,
        "by_type": sorted(by_type, key=lambda x: x["type"]),
        "notes_brief": notes_brief
    }

def main():
    use_real_data = input("Connect to MIMIC-III PostgreSQL database? (y/n): ").lower().startswith('y')
    if not use_real_data:
        return None
    try:
        db_config = MIMICProcessor.get_db_config()
        processor = MIMICProcessor(db_config)
        if not processor.engine:
            return None
    except Exception:
        return None
    default_limit = 8
    try:
        limit = int(input(f"Number of notes to process [{default_limit}]: ") or default_limit)
    except ValueError:
        limit = default_limit
    try:
        linker = BiomedicalEntityLinker(engine=processor.engine)
    except RuntimeError:
        return None
    notes_df = processor.load_mimic_notes(limit=limit)
    results: List[Dict] = []
    for _, row in notes_df.iterrows():
        subject_id = int(row['subject_id'])
        text = row['text']
        category = row.get('category', 'Unknown')
        linked_entities = linker.link_entities(text)
        metrics = evaluate_entity_linking(linked_entities)
        results.append({
            'subject_id': subject_id,
            'category': category,
            'text': text,
            'entities': linked_entities,
            'metrics': metrics
        })
    summary = summarize(results)
    report = {"summary": summary, "model": getattr(linker, "model_name", None), "results": results}
    out_path = "gemma3_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Report saved to: {out_path}")

if __name__ == "__main__":
    main()
