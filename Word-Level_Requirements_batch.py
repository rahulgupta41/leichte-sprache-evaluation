import json
import spacy
import time
import json
import requests
import re
import pandas as pd
import textstat

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

textstat.set_lang('de')

def calc_compound_ratio(doc):
    compounds = [t.text for t in doc if len(t.text) > 10 and "-" not in t.text and t.is_alpha]
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    return round(len(compounds) / total_words * 100, 2) if total_words > 0 else 0

def calc_abbrev_ratio(doc):
    abbrevs = [t.text for t in doc if t.text.isupper() and len(t.text) > 1 and t.is_alpha]
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    return round(len(abbrevs) / total_words * 100, 2) if total_words > 0 else 0

def calc_num_ratio(doc):
    nums = [t.text for t in doc if t.like_num or any(c.isdigit() for c in t.text)]
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    return round(len(nums) / total_words * 100, 2) if total_words > 0 else 0

def calc_terminology_ratio(doc):
    nouns = [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"]]
    return round(len(set(nouns)) / len(nouns), 2) if len(nouns) > 0 else 0

def calc_symbol_ratio(doc):
    syms = [t.text for t in doc if t.pos_ == "SYM" or t.text in ["%", "$", "€", "&"]]
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    return round(len(syms) / total_words * 100, 2) if total_words > 0 else 0

def calc_avg_word_len(doc):
    word_lens = [len(t.text) for t in doc if t.is_alpha]
    return round(sum(word_lens) / len(word_lens), 2) if word_lens else 0

def calc_avg_syll(doc):
    sylls = [textstat.syllable_count(t.text) for t in doc if t.is_alpha]
    return round(sum(sylls) / len(sylls), 2) if sylls else 0

def calc_gender_ratio(doc):
    gender_pattern = r"\b\w+(?:\*innen|:innen|_innen|/innen|Innen)\b"
    gendered = [t.text for t in doc if re.search(gender_pattern, t.text)]
    total_words = len([t for t in doc if not t.is_punct and not t.is_space])
    return round(len(gendered) / total_words * 100, 2) if total_words > 0 else 0

AI_PROMPT = """You are a linguistic analysis assistant.
Your task is to analyze a German text and extract three types of information:
1. Abbreviations  
   - Words written fully or mostly in capital letters (e.g., AI, USA, EU).
   - Include the abbreviation exactly as it appears in the text.
2. Complex Words  
   - German words that are long or difficult to read.
   - Consider a word complex if it has more than 8 characters OR has multiple compound parts.
3. Foreign words   
   - Words that are not German (e.g., English, French, Italian words used inside a German sentence).
   - Return the exact word from the text.

Instructions: 
- Analyze the text carefully. Do not explain anything. 
- Extract the exact requested items as they appear in the text.
- If a category has no items, return an empty list.

Return strictly in this JSON format:
{
  "abbreviations": [],
  "foreign_words": [],
  "complex_words": []
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
        "temperature": 0.2
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
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\dataset_word_level_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

ai_keys = [
    "abbreviations",
    "foreign_words",
    "complex_words"
]

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_abbreviations_present')):
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
    
    df.at[i, 'unhyphenated_long_word_percent_normal'] = calc_compound_ratio(doc_n)
    df.at[i, 'abbreviation_percent_normal'] = calc_abbrev_ratio(doc_n)
    df.at[i, 'number_percent_normal'] = calc_num_ratio(doc_n)
    df.at[i, 'terminology_variety_ratio_normal'] = calc_terminology_ratio(doc_n)
    df.at[i, 'symbol_percent_normal'] = calc_symbol_ratio(doc_n)
    df.at[i, 'avg_word_length_normal'] = calc_avg_word_len(doc_n)
    df.at[i, 'avg_syllables_per_word_normal'] = calc_avg_syll(doc_n)
    df.at[i, 'gender_marker_percent_normal'] = calc_gender_ratio(doc_n)
     
    df.at[i, 'unhyphenated_long_word_percent_simple'] = calc_compound_ratio(doc_s)
    df.at[i, 'abbreviation_percent_simple'] = calc_abbrev_ratio(doc_s)
    df.at[i, 'number_percent_simple'] = calc_num_ratio(doc_s)
    df.at[i, 'terminology_variety_ratio_simple'] = calc_terminology_ratio(doc_s)
    df.at[i, 'symbol_percent_simple'] = calc_symbol_ratio(doc_s)
    df.at[i, 'avg_word_length_simple'] = calc_avg_word_len(doc_s)
    df.at[i, 'avg_syllables_per_word_simple'] = calc_avg_syll(doc_s)
    df.at[i, 'gender_marker_percent_simple'] = calc_gender_ratio(doc_s)
    
    ai_res_n = get_ai_feedback(normal_text)
    for k in ai_keys:
        if ai_res_n.get("API_FAILED"):
            df.at[i, f"ai_normal_{k}_present"] = pd.NA
            df.at[i, f"ai_normal_{k}_count"] = pd.NA
            continue
        words_list_n = ai_res_n.get(k, [])
        if not isinstance(words_list_n, list): words_list_n = [str(words_list_n)]

        present_n = len(words_list_n) > 0

        df.at[i, f"ai_normal_{k}_present"] = int(present_n)
        df.at[i, f"ai_normal_{k}_count"] = len(words_list_n) if present_n else 0
    
    time.sleep(4)
    
    ai_res_s = get_ai_feedback(simple_text)
    for k in ai_keys:
        if ai_res_s.get("API_FAILED"):
            df.at[i, f"ai_simple_{k}_present"] = pd.NA
            df.at[i, f"ai_simple_{k}_count"] = pd.NA
            continue
        words_list_s = ai_res_s.get(k, [])
        if not isinstance(words_list_s, list): words_list_s = [str(words_list_s)]

        present_s = len(words_list_s) > 0

        df.at[i, f"ai_simple_{k}_present"] = int(present_s)
        df.at[i, f"ai_simple_{k}_count"] = len(words_list_s) if present_s else 0
            
    time.sleep(4)        
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(df)} rows... saving progress!")
        df.to_excel(OUTPUT_PATH, index=False)        
print(f"Saving results to {OUTPUT_PATH}...")
df.to_excel(OUTPUT_PATH, index=False)
print("Done!")
