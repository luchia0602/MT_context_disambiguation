!pip install -q datasets transformers accelerate unbabel-comet sentencepiece

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from comet import download_model, load_from_checkpoint

MODEL_ID = "facebook/nllb-200-distilled-600M"

GOLD_TEST_PATH = "gold_en_ja_test.json"

MAX_CONTEXT_TURNS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = NllbTokenizer.from_pretrained(
    MODEL_ID,
    src_lang="eng_Latn",
    tgt_lang="jpn_Jpan"
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()

with open(GOLD_TEST_PATH, "r", encoding="utf-8") as f:
    dialogues = json.load(f)

evaluation_samples = []
forced_bos_token_id = tokenizer.convert_tokens_to_ids("jpn_Jpan")

with torch.no_grad():

    for dialogue in tqdm(dialogues):

        utterances = dialogue["utterances"]

        for idx, utt in enumerate(utterances):

            src = utt["utterance_en"]
            ref = utt["utterance"]

            start = max(0, idx - MAX_CONTEXT_TURNS)
            context = utterances[start:idx]

            context_text = ""
            for c in context:
                context_text += f"{c['speaker']}: {c['utterance_en']}\n"

            full_input = context_text + f"{utt['speaker']}: {src}"

            inputs = tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

            evaluation_samples.append({
                "src": full_input,   
                "mt": pred,          
                "ref": ref           
            })


comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

comet_output = comet_model.predict(
    evaluation_samples,
    batch_size=16,
    gpus=1 if torch.cuda.is_available() else 0
)

print(f"D-COMET (BASELINE NLLB): {comet_output.system_score:.4f}")

for i in range(5):
    if i < len(evaluation_samples):
        s = evaluation_samples[i]
        print(s["src"])
        print(s["ref"])
        print(s["mt"])