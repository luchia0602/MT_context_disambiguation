!pip install -q sudachipy sudachidict-core pykakasi gender-guesser gensim unbabel-comet evaluate datasets "transformers<4.40.0" accelerate "numpy<2.0.0" "pandas<2.2.0" "pyarrow<15.0.0"

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json
import torch
import math
import re
import bz2
import urllib.request
import numpy as np
import pykakasi
import gender_guesser.detector as gender
from tqdm import tqdm

from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer
)

from comet import download_model, load_from_checkpoint

from sudachipy import dictionary
from sudachipy import tokenizer as sudachi_tokenizer
from gensim.models import KeyedVectors

MODEL_NAME = "crowwwwww6/nllb-ja-en-mem-al"

TOPIC1_PATH = "/content/topic1_translated.json"
TOPIC2_PATH = "/content/topic2_translated.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CONTEXT = 4
COMET_BATCH = 16

TRANSLATION_BATCH = 8

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("DEVICE:", DEVICE)

hf_tokenizer = NllbTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    src_lang="jpn_Jpan",
    tgt_lang="eng_Latn"
)

special_tokens = [
    '[NEW_DIALOGUE]', '[MEM:', 'SPK=', 'LST=', 'CTX=',
    'C1_ZSUB=', 'C2_ZSUB=', 'C1_ZOBJ=', 'C2_ZOBJ=', 'LAS=', ']'
]
hf_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

print(f"Loading {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)

model.resize_token_embeddings(len(hf_tokenizer))

model.to(DEVICE)
model.eval()

print("Loading COMET...")
comet = load_from_checkpoint(
    download_model("Unbabel/wmt22-comet-da")
)

def process_single_sentence(ja_speaker: str, ja_sentence: str) -> str:
    processor = RealTimeDialogueProcessor()
    return processor.process_turn(ja_speaker, ja_sentence)

with open(TOPIC1_PATH, "r", encoding="utf-8") as f:
    topic1 = json.load(f)

with open(TOPIC2_PATH, "r", encoding="utf-8") as f:
    topic2 = json.load(f)

dataset = topic1 + topic2
print("TOTAL DIALOGUES:", len(dataset))

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

all_comet_data = []

with torch.no_grad():
    for dialogue in tqdm(dataset):
        processor = RealTimeDialogueProcessor()
        context_history = []

        if "conversation" in dialogue:
            turns = dialogue["conversation"]
            def get_speaker(turn): return turn["ja_speaker"]
            def get_ja(turn): return turn["ja_sentence"]
            def get_en(turn): return turn["en_sentence"]
        else:
            turns = dialogue["utterances"]
            def get_speaker(turn): return turn["speaker"]
            def get_ja(turn): return turn["utterance"]
            def get_en(turn): return turn["utterance_en"]

        prepared_samples = []

        for turn in turns:
            speaker = get_speaker(turn)
            ja_text = get_ja(turn)
            ref_en = get_en(turn)

            tagged = processor.process_turn(
                speaker,
                ja_text
            )

            context_history.append(tagged)

            full_context = "\n".join(
                context_history[-MAX_CONTEXT:]
            )

            prepared_samples.append({
                "src": ja_text,
                "context": full_context,
                "ref": ref_en
            })

        for batch in batched(prepared_samples, TRANSLATION_BATCH):
            contexts = [x["context"] for x in batch]

            inputs = hf_tokenizer(
                contexts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(DEVICE)

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                do_sample=False
            )

            preds = hf_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            for sample, pred_en in zip(batch, preds):
                all_comet_data.append({
                    "src": sample["src"],
                    "mt": pred_en,
                    "ref": sample["ref"]
                })

comet_output = comet.predict(
    all_comet_data,
    batch_size=COMET_BATCH,
    gpus=1 if DEVICE == "cuda" else 0
)

print(comet_output.system_score)