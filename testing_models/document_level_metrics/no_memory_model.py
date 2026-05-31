!pip uninstall -y tensorflow tensorflow-cpu tf-keras jax jaxlib
!pip install -U pip
!pip install numpy==2.1.3 pandas==2.2.2
!pip install evaluate datasets sacrebleu sentencepiece
!pip install protobuf==5.28.3
!pip install "transformers<4.45"
!pip install "pytorch-lightning<2.0"
!pip install unbabel-comet

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import re
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from tqdm import tqdm
from comet import download_model, load_from_checkpoint

tokenizer = NllbTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="jpn_Jpan"
)
model_id = "crowwwwww6/nllb-en-ja-no-mem"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

data_files = {
    "test": "en_ja_test.json"
}

test_dataset = load_dataset("json", data_files=data_files)["test"]

def clean_text(text):
    return re.sub(r"\[.*?\]\s*", "", text).strip()

sources_cleaned = [clean_text(ex["source"]) for ex in test_dataset]
references = [ex["target"] for ex in test_dataset]
predictions = []

batch_size = 16
print(f"Translating {len(sources_cleaned)} sentences (Batch size: {batch_size})...")

with torch.no_grad():
    for i in tqdm(range(0, len(sources_cleaned), batch_size)):
        batch_sources = sources_cleaned[i : i + batch_size]
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

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
comet_data = []

for src, pred, ref in zip(sources_cleaned, predictions, references):
    comet_data.append({
        "src": src,
        "mt": pred,
        "ref": ref
    })

comet_output = comet_model.predict(
    comet_data,
    batch_size=16,
    gpus=1 if torch.cuda.is_available() else 0
)

print(f"COMET Score: {comet_output.system_score:.4f}")

for i in range(5):
    print("SRC:", sources_cleaned[i])
    print("REF:", references[i])
    print("PRED:", predictions[i])