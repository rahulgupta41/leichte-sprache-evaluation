import json
import spacy
import time
import requests
import json
import pandas as pd

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

def calc_valid_length_ratio(doc):
    sents = list(doc.sents)
    return round(sum(1 for s in sents if len([t for t in s if not t.is_punct and not t.is_space]) <= 10) / len(sents) * 100, 1) if sents else 0

def calc_passive_ratio(doc):
    sents = list(doc.sents)
    pass_n = sum(1 for s in sents if "werden" in [t.lemma_.lower() for t in s] and any(t.tag_ in ["VVPP", "VMPP"] for t in s))
    return round(pass_n / len(sents) * 100, 1) if sents else 0

def calc_subordinate_ratio(doc):
    sents = list(doc.sents)
    sub_n = sum(1 for t in doc if t.pos_ == "SCONJ" or t.dep_ in ["mark", "advcl", "ccomp"])
    return round(sub_n / len(sents), 2) if sents else 0

def calc_v2_ratio(doc):
    sents = list(doc.sents)
    pos_list = [next((i+1 for i, t in enumerate(s) if t.tag_ in ["VVFIN", "VAFIN", "VMFIN"]), None) for s in sents]
    valid_pos = [p for p in pos_list if p is not None]
    return round(sum(1 for p in valid_pos if p == 2) / len(valid_pos) * 100, 1) if valid_pos else 0

def calc_cond_ratio(doc):
    sents = list(doc.sents)
    cond_n = sum(1 for t in doc if t.lemma_.lower() in ["wenn", "falls", "sofern"])
    return round(cond_n / len(sents), 2) if sents else 0

def calc_time_count(doc):
    return len([ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]])

def calc_question_count(doc):
    return len([s for s in doc.sents if s.text.strip().endswith("?")])

def calc_shift_ratio(doc):
    addr_n = [t.lemma_.lower() for t in doc if t.tag_ in ["PPER", "PPOSAT"] and t.lemma_.lower() in ["du", "ihr", "sie", "wir", "man"]]
    shifts = sum(a != b for a, b in zip(addr_n, addr_n[1:]))
    return round(shifts / (len(addr_n)-1), 2) if len(addr_n) > 1 else 0

def calc_tense_shift_ratio(doc):
    tenses_n = [t.morph.get("Tense")[0] for t in doc if t.tag_ in ["VVFIN", "VAFIN", "VMFIN"] and t.morph.get("Tense")]
    t_shifts = sum(a != b for a, b in zip(tenses_n, tenses_n[1:]))
    return round(t_shifts / (len(tenses_n)-1), 2) if len(tenses_n) > 1 else 0

AI_PROMPT = """You are a linguistic analysis assistant for German 'Leichte Sprache'.
Your task is to scan the text and find specific syntactic structures.
Do not provide deep explanations. Just check if the specific structure is present, and if yes, extract the exact sentences containing it.

Find the following categories:
1. Time and Date References (Zeit- und Datumsangaben)
   - Extract any sentence that contains a time, date, or duration reference.
2. Questions (Frageform)
   - Extract any sentence that is formatted as a question.
3. Complex Verb Positions (Komplexe Verb-Positionen)
   - Extract any sentence where the verb position is unnatural, highly nested, or violates standard easy-to-read word order.

Instructions:
- Analyze the text carefully.
- Return the exact `sentences` from the text that contain the feature. 
- If a feature is not present, leave the array empty.
- Do not explain why. Return only the requested JSON.

Return the result strictly in this JSON format:
{
  "time_and_date_references": [],
  "questions": [],
  "complex_verb_positions": []
}
TEXT:
<<<
{text_to_analyze}
>>>
"""

def get_ai_feedback(text):
    if pd.isna(text) or not str(text).strip(): return {"API_FAILED": True}
    current_prompt = AI_PROMPT.replace("{text_to_analyze}", str(text).strip())
    
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
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\dataset_syntactic_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

ai_keys = [
    "time_and_date_references",
    "questions",
    "complex_verb_positions"
]

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_time_and_date_references_present')):
        continue

    normal_text = str(row.get("Normal", ""))
    simple_text = str(row.get("Simple", ""))

    # Count Sentences and Words
    _doc_n = nlp_sm(normal_text)
    _doc_s = nlp_sm(simple_text)
    df.at[i, 'num_sentences_normal'] = len(list(_doc_n.sents))
    df.at[i, 'num_sentences_simple'] = len(list(_doc_s.sents))
    df.at[i, 'num_words_normal'] = len([t for t in _doc_n if not t.is_punct and not t.is_space])
    df.at[i, 'num_words_simple'] = len([t for t in _doc_s if not t.is_punct and not t.is_space])

    
    # if pd.isna(normal_text) or pd.isna(simple_text) or normal_text.strip() == "nan" or not normal_text.strip() or not simple_text.strip():
    #     continue

    normal_text_clean = " ".join(normal_text.split())
    simple_text_clean = " ".join(simple_text.split())
    
    doc_n = nlp_sm(normal_text_clean)
    doc_s = nlp_sm(simple_text_clean)

    df.at[i, 'short_sentences_percent_normal'] = calc_valid_length_ratio(doc_n)
    df.at[i, 'passive_sentence_percent_normal'] = calc_passive_ratio(doc_n)
    df.at[i, 'subordinate_markers_per_sent_normal'] = calc_subordinate_ratio(doc_n)
    df.at[i, 'v2_verb_position_percent_normal'] = calc_v2_ratio(doc_n)
    df.at[i, 'conditionals_per_sent_normal'] = calc_cond_ratio(doc_n)
    df.at[i, 'time_refs_count_normal'] = calc_time_count(doc_n)
    df.at[i, 'question_count_normal'] = calc_question_count(doc_n)
    df.at[i, 'perspective_shift_ratio_normal'] = calc_shift_ratio(doc_n)
    df.at[i, 'tense_shift_ratio_normal'] = calc_tense_shift_ratio(doc_n)
     
  
    df.at[i, 'short_sentences_percent_simple'] = calc_valid_length_ratio(doc_s)
    df.at[i, 'passive_sentence_percent_simple'] = calc_passive_ratio(doc_s)
    df.at[i, 'subordinate_markers_per_sent_simple'] = calc_subordinate_ratio(doc_s)
    df.at[i, 'v2_verb_position_percent_simple'] = calc_v2_ratio(doc_s)
    df.at[i, 'conditionals_per_sent_simple'] = calc_cond_ratio(doc_s)
    df.at[i, 'time_refs_count_simple'] = calc_time_count(doc_s)
    df.at[i, 'question_count_simple'] = calc_question_count(doc_s)
    df.at[i, 'perspective_shift_ratio_simple'] = calc_shift_ratio(doc_s)
    df.at[i, 'tense_shift_ratio_simple'] = calc_tense_shift_ratio(doc_s)
    

    ai_res_n = get_ai_feedback(normal_text)
    for k in ai_keys:
        if ai_res_n.get("API_FAILED"):
            df.at[i, f"ai_normal_{k}_present"] = pd.NA
            df.at[i, f"ai_normal_{k}_count"] = pd.NA
            continue
        sents_n = ai_res_n.get(k, [])
        if not isinstance(sents_n, list): sents_n = [str(sents_n)]

        present_n = len(sents_n) > 0

        df.at[i, f"ai_normal_{k}_present"] = int(present_n)
        df.at[i, f"ai_normal_{k}_count"] = len(sents_n) if present_n else 0
    
    time.sleep(4)
    

    ai_res_s = get_ai_feedback(simple_text)
    for k in ai_keys:
        if ai_res_s.get("API_FAILED"):
            df.at[i, f"ai_simple_{k}_present"] = pd.NA
            df.at[i, f"ai_simple_{k}_count"] = pd.NA
            continue
        sents_s = ai_res_s.get(k, [])
        if not isinstance(sents_s, list): sents_s = [str(sents_s)]

        present_s = len(sents_s) > 0

        df.at[i, f"ai_simple_{k}_present"] = int(present_s)
        df.at[i, f"ai_simple_{k}_count"] = len(sents_s) if present_s else 0
            
    time.sleep(4)        
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(df)} rows... saving progress!")
        df.to_excel(OUTPUT_PATH, index=False)    

        
print(f"Saving results to {OUTPUT_PATH}...")
df.to_excel(OUTPUT_PATH, index=False)
print("Done!")
