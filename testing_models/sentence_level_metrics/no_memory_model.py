import os
import re
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

from huggingface_hub import login
MODEL_ID = "crowwwwww6/nllb-ja-en-no-mem-baseline" 
login(token="HF_TOKEN")
TEST_PATH = "/kaggle/input/datasets/liudmilashlyakhtina/mem-thesis/ja_en_test.json"

def strip_mem(text):
    return re.sub(r"^\[MEM:[^\]]*\]\s*", "", text).strip()

def main():
    tokenizer = NllbTokenizer.from_pretrained(MODEL_ID, src_lang="jpn_Jpan", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="cuda:0"
    )

    dataset = load_dataset("json", data_files={"test": TEST_PATH})["test"]

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    predictions = []
    references = []
    
    for i, example in enumerate(tqdm(dataset, desc="Generating Translations")):
        original_src = example["source"]
        ref_text = example["target"]
        clean_src = strip_mem(original_src)
        inputs = tokenizer(clean_src, return_tensors="pt", truncation=True, max_length=128).to("cuda:0")

        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=128, 
                num_beams=4         
            )

        pred_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        predictions.append(pred_text)
        references.append([ref_text]) 

    results_bleu = sacrebleu.compute(predictions=predictions, references=references)
    results_chrf = chrf.compute(predictions=predictions, references=references)

    print(f"SacreBLEU: {results_bleu['score']:.2f}")
    print(f"chrF++:    {results_chrf['score']:.2f}")

if __name__ == "__main__":
    main()