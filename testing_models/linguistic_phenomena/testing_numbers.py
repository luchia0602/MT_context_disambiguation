!pip install transformers sentencepiece spacy datasets
!python -m spacy download en_core_web_sm

import torch
import spacy
import re
import json
from collections import Counter, defaultdict
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

model_id = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(model_id, src_lang="jpn_Jpan")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    return re.sub(r"\[MEM:.*?\]\s*", "", text)

def extract_noun_number(text):
    doc = nlp(text)
    results = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            num = token.morph.get("Number")
            if num:
                results.append((token.lemma_.lower(), num[0])) 

    return results

with open("prof_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

TP = FP = FN = 0

confusion = defaultdict(lambda: defaultdict(int))

ref_all = []
pred_all = []

for example in tqdm(data):
    src = clean_text(example["source"])
    ref = example["target"]

    inputs = tokenizer(src, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=5
        )

    pred = tokenizer.decode(output[0], skip_special_tokens=True)

    ref_pairs = extract_noun_number(ref)
    pred_pairs = extract_noun_number(pred)

    ref_all.extend(ref_pairs)
    pred_all.extend(pred_pairs)

    ref_counter = Counter(ref_pairs)
    pred_counter = Counter(pred_pairs)

    for key in set(ref_counter.keys()) | set(pred_counter.keys()):
        tp = min(ref_counter[key], pred_counter[key])
        fp = max(0, pred_counter[key] - ref_counter[key])
        fn = max(0, ref_counter[key] - pred_counter[key])

        TP += tp
        FP += fp
        FN += fn

    ref_dict = defaultdict(list)
    pred_dict = defaultdict(list)

    for lemma, num in ref_pairs:
        ref_dict[lemma].append(num)

    for lemma, num in pred_pairs:
        pred_dict[lemma].append(num)

    for lemma in ref_dict:
        if lemma in pred_dict:
            for r, p in zip(ref_dict[lemma], pred_dict[lemma]):
                confusion[r][p] += 1

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

ref_counts = Counter(num for _, num in ref_all)
pred_counts = Counter(num for _, num in pred_all)

print("REF number distribution:", dict(ref_counts))
print("PRED number distribution:", dict(pred_counts))

missing = ref_counts - pred_counts
extra = pred_counts - ref_counts

print("Missing numbers:", dict(missing))
print("Hallucinated numbers:", dict(extra))
print("CONFUSION MATRIX (ref → pred):")
for r in confusion:
    print(f"{r} -> {dict(confusion[r])}")