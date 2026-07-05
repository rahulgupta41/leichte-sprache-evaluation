import json
import spacy
import pandas as pd
import time
import requests
import json
from collections import Counter

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

def calc_flow_metrics(text):
  
    if pd.isna(text) or not str(text).strip():
        return 0, 0, 0
    
    text = str(text)
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    doc = nlp_sm(" ".join(text.split()))
    

    words_per_para = [len([t for t in nlp_sm(p) if not t.is_punct]) for p in paras]
    avg_chunk_size = sum(words_per_para) / len(words_per_para) if len(words_per_para) > 0 else 0
    

    verbs_per_para = [sum(1 for t in nlp_sm(p) if t.pos_ in ["VERB", "AUX"]) for p in paras]
    avg_info_density = sum(verbs_per_para) / len(verbs_per_para) if len(verbs_per_para) > 0 else 0

    nouns = [t.lemma_.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]
    redundant = {k: v for k, v in Counter(nouns).items() if v > 1}
    redundancy_ratio = sum(redundant.values()) / len(nouns) if len(nouns) > 0 else 0
    
    return round(avg_chunk_size, 2), round(avg_info_density, 2), round(redundancy_ratio, 2)

AI_PROMPT = """You are a linguistic analysis assistant for German 'Leichte Sprache'.
Your task is to scan the text and evaluate specific "Information Flow" metrics.
Do not provide explanations. Just check if there is a problem/violation of the metric, and if yes, extract the exact sentences where the issue occurs.

Evaluate the following 7 metrics:
1. Topic Progression Analysis (Themenentwicklung)
   - Rule: There should be smooth transitions between topics.
   - Check if there are abrupt, sudden, or confusing topic jumps between sentences or paragraphs.
   - Extract the sentences where a sudden confused topic shift happens.
2. Paragraph Cohesion Measurement (Absatz-Zusammenhalt)
   - Rule: Keep paragraphs internally focused (One main idea per paragraph).
   - Check if any single paragraph is packed with multiple different, unrelated, or conflicting ideas.
   - Extract the sentences that introduce "extra" unrelated ideas into a paragraph.
3. Process Step Separation Effectiveness (Prozess-Schritte Trennung)
   - Rule: Use lists/headings for multi-step procedures.
   - Check if long procedural steps or instructions are written as running text instead of a clear bulleted list.
   - Extract the sentences that should ideally be reformatted into a bullet-point list.
4. Information Hierarchy Clarity (Klare Informations-Hierarchie)
   - Rule: Show main vs. sub-points distinctly.
   - Check if the text fails to clearly distinguish what is the main overarching point versus what are just sub-points or examples.
   - Extract the sentences where sub-points are confusingly presented as main points (or vice versa).
5. Key Point Emphasis Tracking (Hervorhebung der Kernpunkte)
   - Rule: Highlight crucial info succinctly (Don't bury the lead).
   - Check if the absolute most important piece of information is buried randomly in the middle or end of the text, rather than being placed prominently.
   - Extract the crucial sentences that are "buried" and not emphasized properly.
6. Cause-Effect Explicitness (Klare Ursache-Wirkung)
   - Rule: Clearly indicate cause-effect links (using logic words like 'weil', 'deshalb', 'daher') instead of implied causation.
   - Check if there are implied consequences or missing connectors that make it confusing to understand WHY something happened.
   - Extract the sentences where causation is implied but not explicitly stated.
7. Transition Effectiveness (Wirksame Übergänge)
   - Rule: The text should guide readers smoothly between ideas using transitional discourse markers.
   - Check if sentences or paragraphs feel disconnected from each other because of missing transition words.
   - Extract the sentences or paragraphs that feel abruptly disconnected.

Instructions: Include all 7 metric keys. For each, return a boolean `Metric_Valid` and a list of exact `sentences` exhibiting the problem.

Return strictly in this JSON format mapping exactly to the 7 keys:
{
  "topic_progression_analysis": {"Metric_Valid": true, "sentences": []},
  "paragraph_cohesion": {"Metric_Valid": false, "sentences": []},
  "process_step_separation": {"Metric_Valid": false, "sentences": []},
  "information_hierarchy_clarity": {"Metric_Valid": false, "sentences": []},
  "key_point_emphasis": {"Metric_Valid": false, "sentences": []},
  "cause_effect_explicitness": {"Metric_Valid": false, "sentences": []},
  "transition_effectiveness": {"Metric_Valid": false, "sentences": []}
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
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\dataset_information_flow_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

ai_keys = [
    "topic_progression_analysis",
    "paragraph_cohesion",
    "process_step_separation",
    "information_hierarchy_clarity",
    "key_point_emphasis",
    "cause_effect_explicitness",
    "transition_effectiveness"
]

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_topic_progression_analysis_present')):
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

    
    
    if pd.isna(normal_text) or pd.isna(simple_text) or normal_text.strip() == "nan" or not normal_text.strip() or not simple_text.strip():
        continue
        

    n_chunk, n_info, n_red = calc_flow_metrics(normal_text)
    s_chunk, s_info, s_red = calc_flow_metrics(simple_text)
    
    df.at[i, 'avg_chunk_size_normal'] = n_chunk
    df.at[i, 'avg_chunk_size_simple'] = s_chunk
    df.at[i, 'avg_info_density_normal'] = n_info
    df.at[i, 'avg_info_density_simple'] = s_info
    df.at[i, 'redundancy_ratio_normal'] = n_red
    df.at[i, 'redundancy_ratio_simple'] = s_red
    

    
    ai_res_n = get_ai_feedback(normal_text)
    for k in ai_keys:
        data_n = ai_res_n.get(k, {})
        sents_n = data_n.get("sentences", [])

        if not isinstance(sents_n, list): sents_n = [str(sents_n)]

        present_n = len(sents_n) > 0

        df.at[i, f"ai_normal_{k}_present"] = int(present_n)
        df.at[i, f"ai_normal_{k}_count"] = len(sents_n) if present_n else 0
    
    time.sleep(4)
    
    
    ai_res_s = get_ai_feedback(simple_text)
    for k in ai_keys:
        data_s = ai_res_s.get(k, {})
        sents_s = data_s.get("sentences", [])

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
