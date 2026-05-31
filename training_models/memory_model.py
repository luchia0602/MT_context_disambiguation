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

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def print_gpu_memory(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9

def main():
    base_model_id = "facebook/nllb-200-distilled-600M"
    new_repo_id   = "crowwwwww6/nllb-ja-en-memory-v2.0"

    tokenizer = NllbTokenizer.from_pretrained(
        base_model_id,
        src_lang="jpn_Jpan",
        tgt_lang="eng_Latn"
    )

    special_tokens = [
        '[NEW_DIALOGUE]', '[MEM:', 'SPK=', 'LST=', 'CTX=', 
        'C1_ZSUB=', 'C2_ZSUB=', 'C1_ZOBJ=', 'C2_ZOBJ=', 'LAS=', ']'
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

    model.resize_token_embeddings(len(tokenizer))

    data_files = {
        "train":      "/kaggle/input/datasets/liudmilashlyakhtina/mem-thesis-ja-en/ja_en_train.json",
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
        output_dir="./nllb_v2_checkpoints",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        num_train_epochs=3.0,
        fp16=False,
        bf16=False,
        predict_with_generate=False,
        fp16_full_eval=False,
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=1000,
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
        callbacks=[ForcePrintLossCallback(), ClearCacheCallback()]
    )

    torch.cuda.empty_cache()
    trainer.train()

    torch.cuda.empty_cache()

    trainer.args.predict_with_generate = True
    trainer.args.generation_max_length  = 128  

    final_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        metric_key_prefix="final_eval"
    )
    print(f"Final evaluation metrics: {final_metrics}")

    trainer.save_model("./nllb_ja_en_v2_final")
    tokenizer.save_pretrained("./nllb_ja_en_v2_final")
    trainer.push_to_hub("Final V2.0 model upload complete")

    log_history = trainer.state.log_history

    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        elif "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

if __name__ == "__main__":
    main()