import json
from typing import Dict, List
import re

def load_gazetteer_json(file_dir):
    with open(file_dir, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_gazetteer(txt: str, gazetteer: Dict[str, List[ str]]):
    text_lower = txt.lower()
    extracted = {}
    
    for category, terms in gazetteer.items():
        matches = []
        
        for term in terms:
            # Use word boundaries for precise matches
            pattern = r'\b' + re.escape(term) + r'\b'
            matched = re.search(pattern, text_lower)
            if matched:
                matches.append(term)

        if matches:
            extracted[category] = matches
    return extracted