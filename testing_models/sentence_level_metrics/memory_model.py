!pip install datasets transformers accelerate sacrebleu mecab-python3 ipadic tqdm

import torch
from datasets import load_dataset
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from tqdm import tqdm

model_id = "crowwwwww6/nllb-en-ja-memory-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Loading tokenizer and model from {model_id}...")
tokenizer = NllbTokenizer.from_pretrained(model_id, src_lang="eng_Latn", tgt_lang="jpn_Jpan")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to(device)
model.eval()

test_file_path = "/kaggle/input/datasets/liudmilashlyakhtina/en-ja-dataset/en_ja_test.json"
dataset = load_dataset("json", data_files={"test": test_file_path})
test_data = dataset["test"]

sources = [ex["source"] for ex in test_data]
references = [ex["target"] for ex in test_data]
predictions = []

batch_size = 16  
print(f"Generating translations for {len(sources)} sentences (Batch size: {batch_size})...")

with torch.no_grad():
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_sources = sources[i : i + batch_size]
        
        inputs = tokenizer(
            batch_sources, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=192
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=192,
            num_beams=4,
            early_stopping=True
        )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)

with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for src, ref, pred in zip(sources, references, predictions):
        f.write(f"SRC:  {src}\nREF:  {ref}\nPRED: {pred}\n{'-'*40}\n")

bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="ja-mecab")
chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

print(f"Total Sentences Evaluated: {len(predictions)}")
print(f"SacreBLEU Score: {bleu.score:.2f}")
print(f"chrF++ Score:    {chrf.score:.2f}")