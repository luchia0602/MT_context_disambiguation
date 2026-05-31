import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from huggingface_hub import login

login(token="HF_TOKEN")
repo_id = "crowwwwww6/nllb-ja-en-memory"
tokenizer = NllbTokenizer.from_pretrained(repo_id, src_lang="jpn_Jpan")
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

pool_df = pd.read_json("/kaggle/input/datasets/liudmilashlyakhtina/mem-thesis/al_pool.json")
if 'is_annotated' in pool_df.columns:
    unannotated_pool = pool_df[~pool_df['is_annotated']].copy()
else:
    unannotated_pool = pool_df.copy()

sentences_to_score = unannotated_pool.to_dict(orient="records")
scored_sentences = []

for item in tqdm(sentences_to_score):
    source_text = item["source"]
    inputs = tokenizer(source_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=5,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    confidence_score = outputs.sequences_scores.item()
    predicted_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    scored_sentences.append({
        "source": source_text,
        "model_guess": predicted_text,
        "confidence_score": confidence_score,
        "target": "" 
    })

scored_sentences.sort(key=lambda x: x["confidence_score"])
top_100_uncertain = scored_sentences[:100]
output_file = "annotation_batch_1.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(top_100_uncertain, f, ensure_ascii=False, indent=2)

print("The model's lowest confidence score was:", top_100_uncertain[0]["confidence_score"])