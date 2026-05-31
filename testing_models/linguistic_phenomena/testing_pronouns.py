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

PERSONAL_PRONOUNS = {
    "i","me","you","he","him","she","her","it","we","us","they","them"
}

POSSESSIVE_PRONOUNS = {
    "my","your","his","her","its","our","their",
    "mine","yours","hers","ours","theirs"
}

ALL_PRONOUNS = PERSONAL_PRONOUNS | POSSESSIVE_PRONOUNS

def clean_text(text):
    return re.sub(r"\[MEM:.*?\]\s*", "", text)

def extract_pronouns(text):
    doc = nlp(text.lower())
    return [t.text for t in doc if t.text in ALL_PRONOUNS]

def extract_subject_pronouns(text):
    doc = nlp(text.lower())
    return [
        t.text for t in doc
        if t.text in ALL_PRONOUNS and t.dep_ == "nsubj"
    ]

with open("prof_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

TP = FP = FN = 0

confusion = defaultdict(lambda: defaultdict(int))

ref_all = []
pred_all = []

ref_subjects = []
pred_subjects = []

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

    ref_prons = extract_pronouns(ref)
    pred_prons = extract_pronouns(pred)

    ref_subj = extract_subject_pronouns(ref)
    pred_subj = extract_subject_pronouns(pred)

    ref_all.extend(ref_prons)
    pred_all.extend(pred_prons)

    ref_subjects.extend(ref_subj)
    pred_subjects.extend(pred_subj)

    ref_c = Counter(ref_prons)
    pred_c = Counter(pred_prons)

    for p in ALL_PRONOUNS:
        tp = min(ref_c[p], pred_c[p])
        fp = max(0, pred_c[p] - ref_c[p])
        fn = max(0, ref_c[p] - pred_c[p])

        TP += tp
        FP += fp
        FN += fn

    for r, p in zip(ref_prons, pred_prons):
        confusion[r][p] += 1

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

subj_match = sum(1 for r, p in zip(ref_subjects, pred_subjects) if r == p)
subj_total = len(ref_subjects)
subj_acc = subj_match / subj_total if subj_total else 0

print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

print("Subject Pronoun Accuracy:")
print(f"{subj_acc:.4f}")

ref_counts = Counter(ref_all)
pred_counts = Counter(pred_all)

print("Top REF pronouns:")
print(ref_counts.most_common(10))

print("Top PRED pronouns:")
print(pred_counts.most_common(10))

missing = ref_counts - pred_counts
extra = pred_counts - ref_counts

print("Missing pronouns:")
print(dict(missing))

print("Hallucinated pronouns:")
print(dict(extra))

print("CONFUSION MATRIX (ref → pred):")
for r in confusion:
    print(f"{r} -> {dict(confusion[r])}")