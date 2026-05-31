!pip install transformers datasets evaluate sacrebleu sentencepiece
import torch
import evaluate
import re
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from tqdm import tqdm

model_id = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizer.from_pretrained(model_id, src_lang="jpn_Jpan")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

data_files = {"test": "/content/combined_ja_en_test.json"}  
test_dataset = load_dataset("json", data_files=data_files)["test"]
sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def clean_text(text):
    return re.sub(r"\[.*?\]\s*", "", text)

predictions = []
references = []
check = 0
print(f"Translating {len(test_dataset)} sentences")

for example in test_dataset:
    source_text = clean_text(example["source"])
    target_text = example["target"]

    inputs = tokenizer(source_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=5,
            repetition_penalty=1.2,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )

    prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    predictions.append(prediction)
    references.append([target_text]) 
    check += 1
    if check % 100 == 0:
      print("translated: ", check)
bleu = sacrebleu.compute(predictions=predictions, references=references)
chrf_score = chrf.compute(predictions=predictions, references=references)

print(f"SacreBLEU: {bleu['score']:.2f}")
print(f"chrF++: {chrf_score['score']:.2f}")

for i in range(5):
    print("SRC:", clean_text(test_dataset[i]["source"]))
    print("REF:", references[i][0])
    print("PRED:", predictions[i])