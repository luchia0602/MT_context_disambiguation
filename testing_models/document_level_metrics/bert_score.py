!pip install -q sentence-transformers transformers accelerate sentencepiece

import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_id = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizer.from_pretrained(
    model_id,
    src_lang="jpn_Jpan"
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

with open("topic1_translated.json", "r", encoding="utf-8") as f:
    topic1 = json.load(f)
with open("topic2_translated.json", "r", encoding="utf-8") as f:
    topic2 = json.load(f)

topic1_subset = topic1[:400]
topic2_subset = topic2[:100]
dataset = topic1_subset + topic2_subset

def translate_document(japanese_sentences):

    translations = []
    for sentence in japanese_sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "eng_Latn"
            ),
            max_new_tokens=128
        )

        translation = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )[0]

        translations.append(translation)
    return translations

def document_similarity(reference_text, hypothesis_text):

    ref_emb = embedding_model.encode(
        reference_text,
        convert_to_numpy=True
    )

    hyp_emb = embedding_model.encode(
        hypothesis_text,
        convert_to_numpy=True
    )

    score = cosine_similarity(
        [ref_emb],
        [hyp_emb]
    )[0][0]

    return float(score)

scores = []

for dialogue in tqdm(dataset):

    utterances = dialogue["utterances"]

    source_sentences = [
        u["utterance"]
        for u in utterances
    ]

    reference_sentences = [
        u["utterance_en"]
        for u in utterances
    ]

    hypothesis_sentences = translate_document(
        source_sentences
    )

    reference_document = " ".join(
        reference_sentences
    )

    hypothesis_document = " ".join(
        hypothesis_sentences
    )

    score = document_similarity(
        reference_document,
        hypothesis_document
    )

    scores.append(score)

scores = np.array(scores)

print(f"Documents evaluated: {len(scores)}")
print(f"Mean score: {scores.mean():.4f}")
print(f"Median score: {np.median(scores):.4f}")
print(f"Min score: {scores.min():.4f}")
print(f"Max score: {scores.max():.4f}")
print(f"Std: {scores.std():.4f}")