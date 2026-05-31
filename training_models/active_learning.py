import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

login(token="HF_TOKEN")

class ForcePrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            if "loss" in logs:
                print(f"[Step {step}] TRAINING Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"[Step {step}] VALIDATION Loss: {logs['eval_loss']:.4f}\n")

def main():
    base_model_id = "crowwwwww6/nllb-ja-en-memory-v2.0"
    new_repo_id = "crowwwwww6/nllb-ja-en-memory-v2.3-AL1"
    tokenizer = NllbTokenizer.from_pretrained(base_model_id, src_lang="jpn_Jpan", tgt_lang="eng_Latn")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

    model.config.tie_word_embeddings = False
    model.get_input_embeddings().weight.requires_grad  = False
    model.get_output_embeddings().weight.requires_grad = False

    data_files = {
        "train": "./al_cycle_1_train.json", 
        "validation": "/kaggle/input/datasets/liudmilashlyakhtina/mem-thesis-ja-en/ja_en_dev.json" 
    }
    dataset = load_dataset("json", data_files=data_files)

    def tokenize_fn(examples):
        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset.map(tokenize_fn, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./nllb_al1_checkpoints",
        
        learning_rate=2e-5, 

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",

        num_train_epochs=2,

        fp16=False,
        bf16=False,
        predict_with_generate=False,
        fp16_full_eval=False,

        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,

        push_to_hub=True,
        hub_model_id=new_repo_id,
        hub_private_repo=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[ForcePrintLossCallback()]
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model("./nllb_al1_final")
    tokenizer.save_pretrained("./nllb_al1_final")
    trainer.push_to_hub("Final Active Learning Cycle 1 upload")

if __name__ == "__main__":
    main()