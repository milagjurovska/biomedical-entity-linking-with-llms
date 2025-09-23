import pandas as pd
import numpy as np
import ollama
import psycopg2
from sqlalchemy import create_engine, text
import re
import os
from typing import List, Dict, Tuple, Optional
import json
import warnings
import getpass
from collections import defaultdict

warnings.filterwarnings('ignore')


class BiomedicalEntityLinker:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self.setup_model()

        # Define biomedical entity types and sample knowledge base
        self.entity_types = [
            "Medication", "Disease", "Symptom", "Procedure", "Anatomy", "Test", "Other"
        ]

        # Sample knowledge base (in practice, this would be UMLS, SNOMED-CT, etc.)
        self.knowledge_base = self.create_sample_kb()

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
                except Exception as e:
                    if model_names and model_names[0]:
                        self.model_name = model_names[0]
                        print(f"Using available model: {self.model_name}")

            print(f"✓ Model {self.model_name} ready")

        except Exception as e:
            print(f"Ollama setup failed: {e}")
            raise RuntimeError("Ollama not available")

    def create_sample_kb(self):
        """Create a sample knowledge base with medical entities"""
        kb = {
            # Medications
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

            # Diseases/Conditions
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

            # Symptoms
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

            # Procedures
            "echocardiogram": {
                "id": "UMLS:C0013516",
                "name": "Echocardiography",
                "type": "Procedure",
                "aliases": ["echo", "cardiac ultrasound"],
                "definition": "Ultrasound examination of the heart."
            }
        }
        return kb

    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'stop': ['\n\n', '---']
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def extract_entities(self, clinical_text: str, max_length: int = 1000) -> List[Dict]:
        """Extract biomedical entities from clinical text"""
        if len(clinical_text) > max_length:
            clinical_text = clinical_text[:max_length] + "..."

        entity_types_str = ", ".join(self.entity_types)

        prompt = f"""Extract biomedical entities from this clinical text. For each entity, provide the text span and classify it.

Clinical text: "{clinical_text}"

Find entities of these types: {entity_types_str}

Format your response as:
ENTITY: [entity text] | TYPE: [entity type]

Example:
ENTITY: aspirin | TYPE: Medication
ENTITY: chest pain | TYPE: Symptom

Extract entities:"""

        response = self.generate_response(prompt, max_tokens=200)

        # Parse entities from response
        entities = []
        for line in response.split('\n'):
            line = line.strip()
            if 'ENTITY:' in line and 'TYPE:' in line:
                try:
                    parts = line.split('|')
                    entity_part = parts[0].replace('ENTITY:', '').strip()
                    type_part = parts[1].replace('TYPE:', '').strip()

                    if entity_part and type_part:
                        entities.append({
                            'text': entity_part,
                            'type': type_part,
                            'start': clinical_text.lower().find(entity_part.lower()),
                            'end': clinical_text.lower().find(entity_part.lower()) + len(entity_part)
                        })
                except:
                    continue

        return entities

    def find_candidates(self, entity_text: str, entity_type: str = None, top_k: int = 5) -> List[Dict]:
        """Find candidate entities from knowledge base (alias matching)"""
        candidates = []
        entity_lower = entity_text.lower()

        for kb_key, kb_entry in self.knowledge_base.items():
            score = 0

            # Exact match with main name
            if entity_lower == kb_entry['name'].lower():
                score = 100
            # Exact match with alias
            elif entity_lower in [alias.lower() for alias in kb_entry['aliases']]:
                score = 95
            # Partial match with main name
            elif entity_lower in kb_entry['name'].lower() or kb_entry['name'].lower() in entity_lower:
                score = 80
            # Partial match with aliases
            elif any(entity_lower in alias.lower() or alias.lower() in entity_lower
                     for alias in kb_entry['aliases']):
                score = 75

            # Type matching bonus
            if entity_type and entity_type == kb_entry['type']:
                score += 10

            if score > 0:
                candidates.append({
                    'id': kb_entry['id'],
                    'name': kb_entry['name'],
                    'type': kb_entry['type'],
                    'aliases': kb_entry['aliases'],
                    'definition': kb_entry['definition'],
                    'score': score
                })

        # Sort by score and return top-k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    def disambiguate_entity(self, entity_text: str, context: str, candidates: List[Dict]) -> Optional[Dict]:
        """Use LLM to disambiguate between candidate entities (Study 2 approach)"""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Format candidates for LLM
        candidate_info = []
        for i, candidate in enumerate(candidates):
            info = f"{i + 1}. {candidate['name']} ({candidate['id']})\n"
            info += f"   Type: {candidate['type']}\n"
            info += f"   Definition: {candidate['definition']}\n"
            if candidate['aliases']:
                info += f"   Aliases: {', '.join(candidate['aliases'])}\n"
            candidate_info.append(info)

        candidates_text = "\n".join(candidate_info)

        prompt = f"""Given the entity mention "{entity_text}" in the following context, choose the most appropriate entity from the candidates:

Context: "{context}"

Candidates:
{candidates_text}

Which candidate best matches the entity "{entity_text}" in this context? 

Respond with only the number (1, 2, 3, etc.) of the best candidate:"""

        response = self.generate_response(prompt, max_tokens=10)

        # Parse response
        try:
            choice = int(re.search(r'\d+', response).group())
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
        except:
            pass

        # Fallback to highest scoring candidate
        return candidates[0]

    def link_entities(self, clinical_text: str) -> List[Dict]:
        """Complete entity linking pipeline"""
        # Step 1: Extract entities
        entities = self.extract_entities(clinical_text)

        # Step 2: Link each entity
        linked_entities = []
        for entity in entities:
            # Find candidates
            candidates = self.find_candidates(entity['text'], entity['type'])

            # Disambiguate
            best_match = self.disambiguate_entity(
                entity['text'],
                clinical_text[max(0, entity['start'] - 100):entity['end'] + 100],
                candidates
            )

            # Store result
            linked_entity = {
                'mention': entity['text'],
                'type': entity['type'],
                'start': entity['start'],
                'end': entity['end'],
                'linked_entity': best_match,
                'candidates': candidates
            }
            linked_entities.append(linked_entity)

        return linked_entities


class MIMICProcessor:
    def __init__(self, db_config: Dict = None):
        self.db_config = db_config
        self.engine = None

        if db_config:
            self.setup_connection()

    def setup_connection(self):
        try:
            print("Attempting to connect to PostgreSQL database...")

            conn_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"

            self.engine = create_engine(
                conn_string,
                connect_args={"connect_timeout": 10},
                pool_timeout=20
            )

            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            print("✓ Successfully connected to MIMIC-III PostgreSQL database")

        except Exception as e:
            print(f"✗ Failed to connect to PostgreSQL: {e}")
            self.engine = None

    @staticmethod
    def get_db_config() -> Dict:
        print("\n=== MIMIC-III PostgreSQL Configuration ===")

        config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'mimic',
            'user': input("PostgreSQL username: ").strip() or 'postgres',
            'password': getpass.getpass("PostgreSQL password: ")
        }

        return config

    def create_sample_data(self) -> pd.DataFrame:
        sample_notes = [
            {
                'subject_id': 1001,
                'hadm_id': 2001,
                'text': "Patient admitted with chest pain and shortness of breath. Started on metoprolol for hypertension. Aspirin prescribed for cardioprotection. Echocardiogram shows normal function.",
                'category': 'Discharge summary'
            },
            {
                'subject_id': 1002,
                'hadm_id': 2002,
                'text': "Diabetic patient with pneumonia. Pain managed with morphine. Patient has history of hypertension, currently on lisinopril.",
                'category': 'Physician'
            }
        ]
        return pd.DataFrame(sample_notes)

    def load_mimic_notes(self, limit: int = 50) -> pd.DataFrame:
        if not self.engine:
            print("No database connection available. Using sample data.")
            return self.create_sample_data()

        try:
            query = f"""
            SELECT 
                subject_id,
                hadm_id,
                category,
                text
            FROM public.noteevents 
            WHERE category IN ('Discharge summary', 'Physician', 'Nursing')
            AND iserror IS DISTINCT FROM '1'
            AND text IS NOT NULL
            AND LENGTH(text) BETWEEN 200 AND 2000
            ORDER BY subject_id
            LIMIT {limit}
            """

            print(f"Loading {limit} clinical notes for entity linking...")
            df = pd.read_sql_query(query, self.engine)
            print(f"✓ Loaded {len(df)} notes from MIMIC-III")

            df['text'] = df['text'].str.replace(r'\n+', ' ', regex=True)
            df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
            df['text'] = df['text'].str.strip()

            return df

        except Exception as e:
            print(f"✗ Error loading MIMIC-III data: {e}")
            return self.create_sample_data()


def evaluate_entity_linking(predicted_links: List[Dict], ground_truth: List[Dict] = None) -> Dict:
    """Evaluate entity linking performance"""
    if not ground_truth:
        # For demo purposes, just count successful links
        successful_links = sum(1 for link in predicted_links if link['linked_entity'] is not None)
        total_entities = len(predicted_links)

        return {
            'total_entities': total_entities,
            'successful_links': successful_links,
            'linking_accuracy': successful_links / total_entities if total_entities > 0 else 0
        }

    # More sophisticated evaluation would go here
    return {}


def main():
    print("=== MIMIC-III Biomedical Entity Linking with Llama 3.2 ===\n")

    # Database setup
    use_real_data = input("Connect to MIMIC-III PostgreSQL database? (y/n): ").lower().startswith('y')

    if use_real_data:
        try:
            db_config = MIMICProcessor.get_db_config()
            processor = MIMICProcessor(db_config)

            if processor.engine:
                default_limit = 5  # Start small for entity linking
                try:
                    limit = int(input(f"\nNumber of notes to process [{default_limit}]: ") or default_limit)
                except ValueError:
                    limit = default_limit
            else:
                processor = MIMICProcessor()
                limit = 2

        except Exception as e:
            print(f"Database setup failed: {e}")
            processor = MIMICProcessor()
            limit = 2

    else:
        print("Using sample data for demonstration.")
        processor = MIMICProcessor()
        limit = 2

    # Initialize entity linker
    print("\nInitializing Biomedical Entity Linker...")
    try:
        linker = BiomedicalEntityLinker()
    except RuntimeError as e:
        print(f"Failed to initialize entity linker: {e}")
        return None

    # Load data
    print(f"\nLoading clinical notes...")
    notes_df = processor.load_mimic_notes(limit=limit)
    print(f"Loaded {len(notes_df)} notes\n")

    # Process each note
    results = []

    print("Starting biomedical entity linking...\n")

    for idx, row in notes_df.iterrows():
        subject_id = row['subject_id']
        text = row['text']
        category = row.get('category', 'Unknown')

        print(f"Processing Subject {subject_id} - {category}")
        print(f"Text preview: {text[:200]}...")

        # Perform entity linking
        linked_entities = linker.link_entities(text)

        print(f"Found {len(linked_entities)} entities:")
        for entity in linked_entities:
            linked_name = entity['linked_entity']['name'] if entity['linked_entity'] else 'UNLINKED'
            linked_id = entity['linked_entity']['id'] if entity['linked_entity'] else 'N/A'
            print(f"  • {entity['mention']} ({entity['type']}) → {linked_name} ({linked_id})")

        # Store results
        result = {
            'subject_id': subject_id,
            'category': category,
            'text': text,
            'entities': linked_entities,
            'metrics': evaluate_entity_linking(linked_entities)
        }
        results.append(result)

        print(f"  Linking accuracy: {result['metrics']['linking_accuracy']:.2f}")
        print()

    # Overall performance
    print("=== OVERALL RESULTS ===")

    total_entities = sum(r['metrics']['total_entities'] for r in results)
    successful_links = sum(r['metrics']['successful_links'] for r in results)
    overall_accuracy = successful_links / total_entities if total_entities > 0 else 0

    print(f"Total entities found: {total_entities}")
    print(f"Successfully linked: {successful_links}")
    print(f"Overall linking accuracy: {overall_accuracy:.3f}")

    # Entity type breakdown
    entity_types = defaultdict(int)
    linked_types = defaultdict(int)

    for result in results:
        for entity in result['entities']:
            entity_types[entity['type']] += 1
            if entity['linked_entity']:
                linked_types[entity['type']] += 1

    print(f"\nEntity Type Breakdown:")
    for entity_type, count in entity_types.items():
        linked_count = linked_types[entity_type]
        accuracy = linked_count / count if count > 0 else 0
        print(f"  {entity_type}: {linked_count}/{count} ({accuracy:.2f})")

    # Save results
    with open("biomedical_entity_linking_results.json", 'w') as f:
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = {
                'subject_id': int(result['subject_id']),
                'category': result['category'],
                'text': result['text'],
                'entities': result['entities'],
                'metrics': result['metrics']
            }
            json_results.append(json_result)

        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: biomedical_entity_linking_results.json")

    return results


if __name__ == "__main__":
    results = main()