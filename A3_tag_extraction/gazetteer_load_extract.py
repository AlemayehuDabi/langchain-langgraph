import json
from typing import Dict, List, Any
import re
from pathlib import Path

def load_gazetteer_json(file_dir):
    full_path = Path(__file__).resolve().parent / file_dir
    with open(full_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_from_gazetteer(text: str, gazetteer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text_lower = text.lower()
    matches = []

    for entry in gazetteer:
        # this concatnate the two array 
        all_terms = [entry["canonical"]] + entry.get("aliases", [])
        # loop through the array
        for term in all_terms:
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            if re.search(pattern, text_lower):
                matches.append({
                    "id": entry["id"],
                    "matched": term,
                    "canonical": entry["canonical"],
                    "type": entry["type"],
                    "category": entry["metadata"]["category"]
                })
                break

    return matches