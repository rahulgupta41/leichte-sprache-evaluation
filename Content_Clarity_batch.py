import json
import spacy
import pandas as pd
import math
import requests
import time
import json
from collections import Counter

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

def calc_semantic_similarity(doc_n, doc_s):

    core_lemmas_n = [t.lemma_.lower() for t in doc_n if t.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]]
    core_lemmas_s = [t.lemma_.lower() for t in doc_s if t.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"]]
    freq_n, freq_s = Counter(core_lemmas_n), Counter(core_lemmas_s)
    common_words = set(freq_n.keys()).intersection(set(freq_s.keys()))
    dot_product = sum(freq_n[word] * freq_s[word] for word in common_words)
    mag_n = math.sqrt(sum(count**2 for count in freq_n.values()))
    mag_s = math.sqrt(sum(count**2 for count in freq_s.values()))
    return round((dot_product / (mag_n * mag_s)) * 100, 2) if mag_n * mag_s > 0 else 0

def calc_coherence(doc):

    sents = list(doc.sents)
    if not sents: return 0
    nouns_per_sent = [set(t.lemma_ for t in sent if t.pos_ in ["NOUN", "PROPN"]) for sent in sents]
    coherence_links = sum(1 for i in range(1, len(nouns_per_sent)) if len(nouns_per_sent[i].intersection(nouns_per_sent[i-1])) > 0)
    return round(coherence_links / len(sents), 2)

def calc_info_preservation(doc_n, simple_text):

    vital_info_n = set([ent.text for ent in doc_n.ents] + [t.text for t in doc_n if t.pos_ == "PROPN"])
    total = len(vital_info_n)
    if total == 0: return 1.0
    missing_info = [info for info in vital_info_n if info not in simple_text]
    missing = len(missing_info)
    preserved = total - missing
    return round(preserved / total, 2)

def calc_abstract_word_count(doc):

    abstract_suffixes = ("ung", "keit", "heit", "schaft", "tion", "tät", "ismus")
    abstract = [t.text for t in doc if t.pos_ == "NOUN" and t.text.endswith(abstract_suffixes)]
    return len(abstract)

def calc_pronoun_count(doc):
    pronouns = [t for t in doc if t.tag_ in ["PPER", "PDS"]]
    return len(pronouns)

AI_PROMPT = """You are a linguistic analysis assistant for German 'Leichte Sprache'.
You are given an ORIGINAL (Normal) German text and its SIMPLIFIED (Simple) version.
Compare both and evaluate 4 specific "Content Clarity" metrics.
Do not provide explanations. Just check if there is a problem, and extract the exact sentences.
Evaluate the following 4 metrics:
1. Explanation Completeness 
   - Rule: Hard words, technical terms, or domain-specific words in the Simple text must be explained clearly.
   - Check if any technical or difficult word from the Normal text appears in the Simple text WITHOUT being explained.
   - Extract the sentences from the Simple text that contain unexplained complex terms.
2. Domain-Specific Terminology Handling 
   - Rule: Important information from the Original must not disappear in the Simple text.
   - Check if any vital fact, number, name, or term from the Normal text is completely missing in the Simple text.
   - Extract the sentences from the Normal text that contain the vital information that went missing.
3. Metaphor Appropriateness 
   - Rule: Figurative language (metaphors, idioms, expressions like 'Meilenstein', 'aufs Dach steigen' used figuratively) must be removed or replaced with literal language.
   - Check if the Simple text still contains any figurative language, metaphors, or idioms.
   - Extract the sentences from the Simple text that still contain figurative language.
4. Example Relevance and Clarity 
   - Rule: If examples are used in the Simple text, they must genuinely help the reader understand.
   - Check if any examples in the Simple text are confusing, irrelevant, or unnecessary.
   - Extract the sentences from the Simple text that contain weak, irrelevant, or confusing examples.
5. Information Preservation Validation 
   - Rule: Important information (names, numbers, entities, facts) from the Normal text must not be lost in the Simple text.
   - Check if any key entity, number, or important concept from the Normal text is missing in the Simple text.
   - Extract the sentences from the Normal text that contain important information that is missing in the Simple text.
Instructions:
- For each metric, return a boolean `problem_present` (true = problem found, false = ok).
- If `problem_present` is true, provide the exact `sentences` that exhibit the problem.
- Return only the requested JSON. No explanations.
Return strictly in this JSON format for all 5 metrics:
{
  "explanation_completeness": { "problem_present": false, "problem_count": 0, "sentences": [] },
  "domain_terminology_handling": { "problem_present": false, "missing_item_count": 0, "missing_items": [], "sentences": [] },
  "metaphor_appropriateness": { "problem_present": false, "problem_count": 0, "sentences": [] },
  "example_relevance": { "problem_present": false, "problem_count": 0, "sentences": [] },
  "information_preservation_validation": { "problem_present": false, "missing_info_count": 0, "missing_items": [], "sentences": [] }
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
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\dataset_with_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

ai_metric_map = {
    "explanation_completeness": "ai_explanation_completeness",
    "domain_terminology_handling": "ai_domain_terminology",
    "metaphor_appropriateness": "ai_metaphor_appropriateness",
    "example_relevance": "ai_example_relevance",
    "information_preservation_validation": "ai_information_preservation"
}

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_ambiguity_detection_present')):
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

    

    # if not normal_text or not simple_text:
    #     continue

    doc_n, doc_s = nlp_sm(normal_text), nlp_sm(simple_text)

    df.at[i, 'semantic_similarity'] = calc_semantic_similarity(doc_n, doc_s)
    df.at[i, 'coherence_normal'] = calc_coherence(doc_n)
    df.at[i, 'coherence_simple'] = calc_coherence(doc_s)
    df.at[i, 'information_preservation_score'] = calc_info_preservation(doc_n, simple_text)
    df.at[i, 'vocabulary_concreteness_measurement_normal'] = calc_abstract_word_count(doc_n)
    df.at[i, 'vocabulary_concreteness_measurement_simple'] = calc_abstract_word_count(doc_s)
    df.at[i, 'ambiguity_detection_normal'] = calc_pronoun_count(doc_n)
    df.at[i, 'ambiguity_detection_simple'] = calc_pronoun_count(doc_s)

    ai_results = get_ai_feedback(normal_text, simple_text)
    
    for ai_key, prefix in ai_metric_map.items():
        if ai_results.get("API_FAILED"):
            df.at[i, f"{prefix}_present"] = pd.NA
            df.at[i, f"{prefix}_count"] = pd.NA
            continue
        metric_data = ai_results.get(ai_key, {})
        df.at[i, f"{prefix}_present"] = int(metric_data.get("problem_present", False))
        
        count_val = metric_data.get("problem_count") or metric_data.get("missing_item_count") or metric_data.get("missing_info_count") or 0
        df.at[i, f"{prefix}_count"] = int(count_val) if str(count_val).isdigit() else 0

    time.sleep(4)        
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(df)} rows...")

print(f"Saving results to {OUTPUT_PATH}...")
df.to_excel(OUTPUT_PATH, index=False)
print("Done!")
