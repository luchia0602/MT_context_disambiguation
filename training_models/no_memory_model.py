!pip install datasets transformers accelerate bitsandbytes matplotlib huggingface_hub

import os
import re
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
    model_id = "facebook/nllb-200-distilled-600M"

    tokenizer = NllbTokenizer.from_pretrained(
        model_id,
        src_lang="eng_Latn",
        tgt_lang="jpn_Jpan"
    )
    
    special_tokens = [
        '[NEW_DIALOGUE]', '[MEM:', 'SPK=', 'LST=', 'CTX=', 
        'C1_ZSUB=', 'C2_ZSUB=', 'C1_ZOBJ=', 'C2_ZOBJ=', 'LAS=', ']'
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    data_files = {
        "train": "/kaggle/input/datasets/liudmilashlyakhtina/mem_thesis_en_ja/en_ja_train.json",
        "validation": "/kaggle/input/datasets/liudmilashlyakhtina/mem_thesis_en_ja/en_ja_dev.json"
    }
    dataset = load_dataset("json", data_files=data_files)

    def preprocess(examples):
        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            truncation=True,
            max_length=192
        )
        
    tokenized_datasets = dataset.map(preprocess, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./nllb_en_ja_no_mem_checkpoints",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        weight_decay=0.01,
        num_train_epochs=3,
        fp16=True,
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=300,
        logging_strategy="steps",
        logging_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        push_to_hub=True,
        hub_model_id="crowwwwww6/nllb-en-ja-no-mem-v1",
        hub_strategy="every_save",
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

    trainer.train()

    trainer.push_to_hub("Training complete")

    log_history = trainer.state.log_history

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        elif "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

if __name__ == "__main__":
    main()