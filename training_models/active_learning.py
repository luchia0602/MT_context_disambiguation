import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

login(token="HF_TOKEN")
BASE_MODEL_ID  = "crowwwwww6/nllb-ja-en-memory-v2.0"
NEW_REPO_ID = "crowwwwww6/nllb-ja-en-memory-v2.1-lora"

VAL_PATH = "/kaggle/input/datasets/niranpruksamanee/mem-thesis/combined_ja_en_dev.json"
ACTIVE_PATH = "/kaggle/input/datasets/niranpruksamanee/mem-thesis/batch_1.json"

OVERSAMPLE_FACTOR = 3
SEED = 42

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

def load_active_batch(path):
    raw = load_dataset("json", data_files={"active": path})["active"]

    cleaned = []
    skipped = 0
    for ex in raw:
        src = ex.get("source", "").strip()
        tgt = ex.get("model_guess", "").strip()

        if not src or not tgt:
            skipped += 1
            continue
        cleaned.append({"source": src, "target": tgt})

    print(f"{len(cleaned)} usable examples")
    return cleaned

def prepare_dataset(active_examples, oversample_factor, seed):
    random.seed(seed)
    oversampled = active_examples * oversample_factor
    dataset = Dataset.from_list(oversampled)
    dataset = dataset.shuffle(seed=seed)

    print(f"Active examples (oversampled {oversample_factor}×): {len(oversampled)}")
    print(f"Total training examples: {len(dataset)}")

    return dataset


def main():
    tokenizer = NllbTokenizer.from_pretrained(
        BASE_MODEL_ID,
        src_lang="jpn_Jpan",
        tgt_lang="eng_Latn"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    val_dataset = load_dataset("json", data_files={"validation": VAL_PATH})["validation"]

    active_batch = load_active_batch(ACTIVE_PATH)

    train_raw = prepare_dataset(
        active_examples=active_batch,
        oversample_factor=OVERSAMPLE_FACTOR,
        seed=SEED
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            truncation=True,
            max_length=128
        )

    train_tokenized = train_raw.map(tokenize_fn, batched=True)
    val_tokenized   = val_dataset.map(tokenize_fn, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./nllb_lora_checkpoints",

        learning_rate=1e-4,
        warmup_ratio=0.2,
        weight_decay=0.01,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",

        num_train_epochs=3,

        fp16=False,
        bf16=False,
        predict_with_generate=False,
        fp16_full_eval=False,

        eval_strategy="steps",
        eval_steps=10,
        logging_strategy="steps",
        logging_steps=10,

        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        push_to_hub=True,
        hub_model_id=NEW_REPO_ID,
        hub_private_repo=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            ForcePrintLossCallback(),
            ClearCacheCallback(),
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )

    print_gpu_memory("Pre-train")
    torch.cuda.empty_cache()

    trainer.train()

    torch.cuda.empty_cache()
    print_gpu_memory("Pre-final-eval")

    trainer.args.predict_with_generate = True
    trainer.args.generation_max_length  = 128

    final_metrics = trainer.evaluate(
        eval_dataset=val_tokenized,
        metric_key_prefix="final_eval"
    )
    print(f"Final evaluation metrics: {final_metrics}")

    merged_model = model.merge_and_unload()

    merged_model.save_pretrained("./nllb_ja_en_v2_1_merged")
    tokenizer.save_pretrained("./nllb_ja_en_v2_1_merged")

    merged_model.push_to_hub(NEW_REPO_ID, private=True)
    tokenizer.push_to_hub(NEW_REPO_ID, private=True)
    print(f"Merged model pushed to Hub: {NEW_REPO_ID}")

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