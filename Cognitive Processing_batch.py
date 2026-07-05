import json
import spacy
import pandas as pd
import time
import requests
import json

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

def calc_working_memory_load(doc):
    sents = list(doc.sents)
    if not sents: return 0.0
    verbs_per_sent = [sum(1 for t in s if t.pos_ in ["VERB", "AUX"] and t.tag_ in ["VVFIN", "VAFIN", "VMFIN"]) for s in sents]
    wml_avg = sum(verbs_per_sent) / len(verbs_per_sent)
    return round(wml_avg, 2)

def calc_entity_density(doc):
    sents = list(doc.sents)
    if not sents: return 0.0
    entity_total = sum(len(s.ents) for s in sents)
    density = entity_total / len(sents)
    return round(density, 2)

def calc_sequential_info_processing(doc):
    seq_words = [
        "erstens","zweitens","drittens","viertens","fünftens",
        "zuerst","zunächst","anfangs",
        "dann","danach","daraufhin","anschließend",
        "später",
        "zuletzt","schließlich","abschließend",
        "am ende","zum schluss"
    ]
    return len([t for t in doc if t.lemma_.lower() in seq_words])

def calc_decision_point_complexity(doc):
    sents = list(doc.sents)
    if not sents: return 0.0
    conds_per_sent = [sum(1 for t in s if t.lemma_.lower() in ["wenn", "falls", "sofern"]) for s in sents]
    nested = sum(1 for c in conds_per_sent if c > 1)
    return round(nested / len(sents), 2)

def calc_reference_distance(doc):
    words = [t for t in doc if not t.is_space and not t.is_punct]
    ents = [i for i, t in enumerate(words) if t.pos_ in ["NOUN", "PROPN"]]
    prons = [i for i, t in enumerate(words) if t.tag_ in ["PPER", "PDS"] and t.text.lower() not in ["sie", "ihr", "ihnen"]]
    ref_dist = [p - max(e for e in ents if e < p) for p in prons if any(e < p for e in ents)]
    if not ref_dist: return 0.0
    return round(sum(ref_dist) / len(ref_dist), 2)

AI_PROMPT = """You are a linguistic expert in German.

Analyze BOTH texts separately: NORMAL TEXT and SIMPLE TEXT.

Evaluate these 2 Cognitive Processing metrics for EACH text:

1. Background Knowledge Assumptions
- Check if the text introduces a concept, event, name, organization, or technical term without explaining it clearly.
- If yes, mark problem_present as true and list the exact sentences.

2. Inference Requirement
- Check if the reader must infer meaning, connection, result, or significance that is not stated directly.
- If yes, mark problem_present as true and list the exact sentences.

Return ONLY valid JSON. No explanation. No markdown.

Use exactly this JSON format:
{
  "normal": {
    "background_knowledge_assumptions": { "problem_present": false, "sentences": [] },
    "inference_requirement": { "problem_present": false, "sentences": [] }
  },
  "simple": {
    "background_knowledge_assumptions": { "problem_present": false, "sentences": [] },
    "inference_requirement": { "problem_present": false, "sentences": [] }
  }
}

NORMAL TEXT:
<<<
{normal_text}
>>>

SIMPLE TEXT:
<<<
{simple_text}
>>>
"""

def get_ai_feedback(normal_text, simple_text):
    current_prompt = AI_PROMPT.replace("{normal_text}", normal_text.strip()).replace("{simple_text}", simple_text.strip())
    
    H2_API_KEY = "sk-1234"  
    URL = "https://ai.h2.de/llm/v1/chat/completions"
    HEADERS = {
        "Authorization": f"Bearer {H2_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "Mistral-3.2",
        "messages": [{"role": "user", "content": current_prompt}],
        "temperature": 0.1
    }
    
    max_retries = 25
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
            
            if not r.ok:
                print(f"API Error (Attempt {attempt+1}/{max_retries}):", r.text[:100].replace('\n', ' '))
                if r.status_code >= 500 or r.status_code == 429 or r.status_code == 504:
                    print("Server overloaded. Waiting 15 seconds before retrying...")
                    import time
                    time.sleep(15)
                    continue
                else:
                    return {"API_FAILED": True}
                
            text_response = r.json()["choices"][0]["message"]["content"].strip()
            text_response = text_response.replace("```json", "").replace("```", "").strip()
            
            start = text_response.find("{")
            end = text_response.rfind("}") + 1
            
            if start != -1 and end != 0:
                json_text = text_response[start:end]
                return json.loads(json_text)
            else:
                print(f"AI returned invalid JSON. Retrying in 15s...")
                import time
                time.sleep(15)
                continue
                
        except json.JSONDecodeError as e:
            print(f"AI returned malformed JSON ({e}). Skipping row.")
            return {"API_FAILED": True}
        except Exception as e:
            print(f"Network Error: {str(e)[:100]}")
            print("Waiting 15 seconds before retrying...")
            import time
            time.sleep(15)
            
    return {"API_FAILED": True}

INPUT_PATH = r"C:\Users\rahul\leichte sprache Code\data.xlsx"
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\cognitive_dataset_with_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_cognitive_load_present')):
        continue

    normal_text = str(row.get("Normal", "")).strip()
    simple_text = str(row.get("Simple", "")).strip()

    # Count Sentences and Words
    _doc_n = nlp_sm(normal_text)
    _doc_s = nlp_sm(simple_text)
    df.at[i, 'num_sentences_normal'] = len(list(_doc_n.sents))
    df.at[i, 'num_sentences_simple'] = len(list(_doc_s.sents))
    df.at[i, 'num_words_normal'] = len([t for t in _doc_n if not t.is_punct and not t.is_space])
    df.at[i, 'num_words_simple'] = len([t for t in _doc_s if not t.is_punct and not t.is_space])

    

    # if not normal_text or not simple_text or normal_text == "nan" or simple_text == "nan":
    #     continue
        
    doc_n = nlp_sm(normal_text)
    doc_s = nlp_sm(simple_text)
    

    df.at[i, 'wml_avg_normal'] = calc_working_memory_load(doc_n)
    df.at[i, 'wml_avg_simple'] = calc_working_memory_load(doc_s)
    
    df.at[i, 'entity_density_normal'] = calc_entity_density(doc_n)
    df.at[i, 'entity_density_simple'] = calc_entity_density(doc_s)
    
    df.at[i, 'seq_per_sentence_normal'] = calc_sequential_info_processing(doc_n)
    df.at[i, 'seq_per_sentence_simple'] = calc_sequential_info_processing(doc_s)
    
    df.at[i, 'decision_complexity_normal'] = calc_decision_point_complexity(doc_n)
    df.at[i, 'decision_complexity_simple'] = calc_decision_point_complexity(doc_s)
    
    df.at[i, 'ref_distance_normal'] = calc_reference_distance(doc_n)
    df.at[i, 'ref_distance_simple'] = calc_reference_distance(doc_s)
    

    ai_results = get_ai_feedback(normal_text, simple_text)
    
    for text_type in ['normal', 'simple']:
        if ai_results.get("API_FAILED"):
            df.at[i, f'ai_bg_knowledge_{text_type}_present'] = pd.NA
            df.at[i, f'ai_bg_knowledge_{text_type}_count'] = pd.NA
            df.at[i, f'ai_inference_{text_type}_present'] = pd.NA
            df.at[i, f'ai_inference_{text_type}_count'] = pd.NA
            continue
        res = ai_results.get(text_type, {})

        bg = res.get("background_knowledge_assumptions", {})
        bg_present = bool(bg.get("problem_present", False))
        bg_sentences = bg.get("sentences", [])
        if not isinstance(bg_sentences, list): bg_sentences = [str(bg_sentences)]
        
        df.at[i, f'ai_bg_knowledge_{text_type}_present'] = int(bg_present)
        df.at[i, f'ai_bg_knowledge_{text_type}_count'] = len(bg_sentences) if bg_present else 0

        inf = res.get("inference_requirement", {})
        inf_present = bool(inf.get("problem_present", False))
        inf_sentences = inf.get("sentences", [])
        if not isinstance(inf_sentences, list): inf_sentences = [str(inf_sentences)]

        df.at[i, f'ai_inference_{text_type}_present'] = int(inf_present)
        df.at[i, f'ai_inference_{text_type}_count'] = len(inf_sentences) if inf_present else 0
    

    time.sleep(4)        
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(df)} rows... saving progress!")
        df.to_excel(OUTPUT_PATH, index=False)        

print(f"Saving results to {OUTPUT_PATH}...")
df.to_excel(OUTPUT_PATH, index=False)
print("Done!")
