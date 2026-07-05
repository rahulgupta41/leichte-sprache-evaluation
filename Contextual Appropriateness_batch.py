import json
import spacy
import pandas as pd
import time
import requests
import json
import textstat

try:
    nlp_sm = spacy.load("de_core_news_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("de_core_news_sm")
    nlp_sm = spacy.load("de_core_news_sm")

textstat.set_lang('de')

def calc_cultural_entity_count(doc):
    return len([ent.text for ent in doc.ents if ent.label_ in ["LOC", "ORG", "MISC"]])

def calc_flesch_reading_ease(text):
    return round(textstat.flesch_reading_ease(text), 2)

def calc_wiener_sachtextformel(text):
    w1 = round(textstat.wiener_sachtextformel(text, 1), 2)
    w2 = round(textstat.wiener_sachtextformel(text, 2), 2)
    w3 = round(textstat.wiener_sachtextformel(text, 3), 2)
    w4 = round(textstat.wiener_sachtextformel(text, 4), 2)
    return w1, w2, w3, w4

AI_PROMPT = """You are a linguistic analysis assistant for German 'Leichte Sprache'.
Your task is to scan the text and evaluate 5 specific "Contextual Appropriateness" metrics.
Do not provide explanations. Just check if there is a problem, and if yes, extract the exact sentences where the issue occurs.

Evaluate the following 5 metrics:
1. Idiomatic Expression Detection (Redewendungen-Erkennung)
   - Rule: Idioms, proverbs, and figurative expressions must be completely removed or replaced with literal language.
   - Check if the text contains any idiomatic expressions, proverbs, or figurative phrases (e.g., 'aufs Dach steigen', 'ins Auge fallen').
   - Extract the sentences that still contain idiomatic or figurative language.
2. Context-Specific Explanation Depth (Kontextuelle Erklärungstiefe)
   - Rule: Domain concepts, event names, or cultural terms should be explained in enough depth for someone unfamiliar with the topic.
   - Check if any specific event, concept, or cultural term is mentioned but given only a surface-level or no explanation.
   - Extract the sentences where the explanation depth is insufficient.
3. Local Reference Appropriateness (Lokale Bezüge-Eignung)
   - Rule: Local place names, organizations, or regional cultural concepts must be briefly explained for readers who are not from that region.
   - Check if local places, events, or organizations are mentioned without explaining what they are or where they are.
   - Extract the sentences that contain unexplained local references.
4. Prior Knowledge Requirements Assessment (Vorwissen-Bewertung)
   - Rule: The reader should need zero prior knowledge to understand the text.
   - Check if the text uses jargon, technical terms, or assumes the reader already knows specific background information.
   - Extract the sentences that assume unexplained prior knowledge or use unexplained jargon.
5. Sociolinguistic Appropriateness Metrics (Soziolinguistische Angemessenheit)
   - Rule: Language must be neutral, respectful, inclusive, and completely free from discrimination.
   - Check if any sentence uses language that is disrespectful, stereotyping, exclusive, or discriminatory toward any group.
   - Extract the sentences that violate sociolinguistic appropriateness.
6. Concept Abstraction Level (Konzept-Abstraktionsgrad)
   - Rule: Prefer concrete, tangible language over abstract concepts. Avoid abstract German nouns, especially those ending in '-ung', '-keit', '-heit', '-schaft', '-tion', '-tät', or '-ismus'.
   - Check if the text contains abstract nouns with these suffixes that make ideas harder to visualize or understand.
   - Extract the sentences that contain such hard, abstract nouns.
7. Cultural Reference Accessibility (Kulturelle Bezüge)
   - Rule: Minimize locale-specific references (like places or cultural events) or ensure they are explained contextually.
   - Check if the text contains unexplained or vague cultural references, locations, or organizations that a stranger would not understand.
   - Extract the sentences that contain unexplained cultural references.

Instructions:
- Analyze the text carefully based on 'Leichte Sprache' guidelines.
- For each metric, return a boolean `problem_present` (true = problem found, false = ok).
- If `problem_present` is true, provide the exact `sentences` from the text that exhibit the problem.
- If `problem_present` is false, leave `sentences` as an empty list.
- Return only the requested JSON. No explanations.

Return strictly in this JSON format for all 5 metrics:
{
  "idiomatic_expression_detection": {"problem_present": false, "sentences": []},
  "context_specific_explanation_depth": {"problem_present": false, "sentences": []},
  "local_reference_appropriateness": {"problem_present": false, "sentences": []},
  "prior_knowledge_requirements": {"problem_present": false, "sentences": []},
  "sociolinguistic_appropriateness": {"problem_present": false, "sentences": []},
  "concept_abstraction_level": {"problem_present": false, "sentences": []},
  "cultural_reference_accessibility": {"problem_present": false, "sentences": []}
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
OUTPUT_PATH = r"C:\Users\rahul\leichte sprache batch\dataset_contextual_metrics.xlsx"

import os
print(f"Loading dataset...")
if os.path.exists(OUTPUT_PATH):
    print(f"Found existing progress! Resuming from {OUTPUT_PATH}")
    df = pd.read_excel(OUTPUT_PATH)
else:
    df = pd.read_excel(INPUT_PATH)

ai_keys = [
    "idiomatic_expression_detection",
    "context_specific_explanation_depth",
    "local_reference_appropriateness",
    "prior_knowledge_requirements",
    "sociolinguistic_appropriateness",
    "concept_abstraction_level",
    "cultural_reference_accessibility"
]

print(f"Processing {len(df)} rows...")

for i, row in df.iterrows():
    # SKIP row if it has already been processed (Resume logic)
    if pd.notna(row.get('ai_normal_idiomatic_expression_detection_present')):
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
        
    doc_n, doc_s = nlp_sm(normal_text), nlp_sm(simple_text)

    df.at[i, 'cultural_entity_count_normal'] = calc_cultural_entity_count(doc_n)
    df.at[i, 'cultural_entity_count_simple'] = calc_cultural_entity_count(doc_s)
    
    df.at[i, 'flesch_reading_ease_normal'] = calc_flesch_reading_ease(normal_text)
    df.at[i, 'flesch_reading_ease_simple'] = calc_flesch_reading_ease(simple_text)
    
    w_n = calc_wiener_sachtextformel(normal_text)
    df.at[i, 'wiener_sachtextformel_1_normal'] = w_n[0]
    df.at[i, 'wiener_sachtextformel_2_normal'] = w_n[1]
    df.at[i, 'wiener_sachtextformel_3_normal'] = w_n[2]
    df.at[i, 'wiener_sachtextformel_4_normal'] = w_n[3]

    w_s = calc_wiener_sachtextformel(simple_text)
    df.at[i, 'wiener_sachtextformel_1_simple'] = w_s[0]
    df.at[i, 'wiener_sachtextformel_2_simple'] = w_s[1]
    df.at[i, 'wiener_sachtextformel_3_simple'] = w_s[2]
    df.at[i, 'wiener_sachtextformel_4_simple'] = w_s[3]

    ai_res_n = get_ai_feedback(normal_text)
    for k in ai_keys:
        data_n = ai_res_n.get(k, {})
        present_n = bool(data_n.get("problem_present", False))
        
        sents_n = data_n.get("sentences", [])
        if not isinstance(sents_n, list): sents_n = [str(sents_n)]

        df.at[i, f"ai_normal_{k}_present"] = int(present_n)
        df.at[i, f"ai_normal_{k}_count"] = len(sents_n) if present_n else 0
        
    time.sleep(4) 
    

    ai_res_s = get_ai_feedback(simple_text)
    for k in ai_keys:
        data_s = ai_res_s.get(k, {})
        present_s = bool(data_s.get("problem_present", False))
        
        sents_s = data_s.get("sentences", [])
        if not isinstance(sents_s, list): sents_s = [str(sents_s)]

        df.at[i, f"ai_simple_{k}_present"] = int(present_s)
        df.at[i, f"ai_simple_{k}_count"] = len(sents_s) if present_s else 0  
           
    time.sleep(4)        
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(df)} rows... saving progress!")
        df.to_excel(OUTPUT_PATH, index=False)        

print(f"Saving results to {OUTPUT_PATH}...")
df.to_excel(OUTPUT_PATH, index=False)
print("Done!")
