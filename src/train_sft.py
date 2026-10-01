"""
train_sft.py
Fine-tunes an open-weight LLM using LoRA or QLoRA on the Fintech Support Triage dataset.
Tracks training time, GPU memory usage, and saves adapter checkpoints and training metrics.
"""

import os
import sys
import time
import json
import yaml
import argparse

# Optimize CUDA allocator for Windows GPU memory stability
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

SYSTEM_PROMPT = (
    "You are the AI Triage Assistant for NeoBank.\n"
    "Your role is to triage customer support messages according to bank policies.\n"
    "You must output a single valid JSON object with the following fields:\n"
    "- \"category\": one of [\"card_payment\", \"card_declined\", \"transfer_delay\", \"refund_status\", \"account_access\", \"suspected_fraud\", \"general_question\"]\n"
    "- \"escalate\": boolean (true if the issue requires human agent intervention or account verification, false otherwise)\n"
    "- \"response\": a concise, policy-grounded response guiding the customer without requesting credentials or making unauthorized financial guarantees.\n\n"
    "Never ask for PIN, password, OTP, or CVV. Output JSON only."
)

def format_example(row, tokenizer):
    assistant_json = json.dumps({
        "category": row["category"],
        "escalate": row["escalate"],
        "response": row["response"]
    }, ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["message"]},
        {"role": "assistant", "content": assistant_json}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def load_jsonl_dataset(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Fintech Support Triage")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else config.get("num_train_epochs", 3)
    out_dir = args.output_dir if args.output_dir is not None else config.get("output_dir", "finetune")

    print("=" * 60)
    print(f"Starting Fine-Tuning: {out_dir}")
    print(f"Base Model: {config['model_name_or_path']}")
    print(f"Target Epochs: {epochs}")
    print(f"Quantization (4-bit): {config.get('load_in_4bit', False)}")
    print(f"LoRA Rank (r): {config.get('lora_r', 16)}, Alpha: {config.get('lora_alpha', 32)}")
    print("=" * 60)

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        initial_mem_mb = 0

    start_time = time.time()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Prepare Datasets
    train_records = load_jsonl_dataset(config["train_data_path"])
    val_records = load_jsonl_dataset(config["val_data_path"])

    print(f"Loaded {len(train_records)} train records, {len(val_records)} val records.")

    def tokenize_function(example):
        full_text = format_example(example, tokenizer)
        tokenized = tokenizer(
            full_text,
            max_length=config.get("max_seq_length", 512),
            truncation=True,
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_ds = Dataset.from_list(train_records).map(tokenize_function, remove_columns=["source", "intent", "message", "category", "escalate", "response", "id"])
    val_ds = Dataset.from_list(val_records).map(tokenize_function, remove_columns=["source", "intent", "message", "category", "escalate", "response", "id"])

    # 3. Model Loading & Quantization
    compute_dtype = torch.float16 if config.get("fp16", True) else torch.float32
    
    bnb_config = None
    if config.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype
        )

    print("Loading base model...")
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        **model_kwargs
    )

    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Setup
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        bias=config.get("bias", "none")
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        num_train_epochs=epochs,
        logging_steps=config.get("logging_steps", 10),
        eval_strategy="epoch" if epochs <= 10 else "steps",
        eval_steps=50 if epochs > 10 else None,
        save_strategy="epoch" if epochs <= 10 else "steps",
        save_steps=100 if epochs > 10 else None,
        save_total_limit=1,
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=config.get("fp16", True),
        bf16=config.get("bf16", False),
        disable_tqdm=False,
        report_to="none",
        seed=config.get("seed", 42)
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print(">>> Starting Training...")
    train_result = trainer.train()

    end_time = time.time()
    total_time_sec = end_time - start_time
    
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem_mb = 0

    print(f"\nTraining Completed!")
    print(f"Total Time: {total_time_sec:.2f} seconds ({total_time_sec / 60:.2f} minutes)")
    print(f"Peak GPU VRAM Allocated: {peak_mem_mb:.2f} MB")

    # Save Model & Tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Adapter and tokenizer saved to {out_dir}")

    # Save Training Metrics
    os.makedirs("outputs/metrics", exist_ok=True)
    exp_name = os.path.basename(out_dir)
    metrics_path = os.path.join("outputs", "metrics", f"{exp_name}_train_metrics.json")

    metrics_data = {
        "experiment": exp_name,
        "base_model": config["model_name_or_path"],
        "quantization": "4-bit (QLoRA)" if config.get("load_in_4bit", False) else "fp16 (LoRA)",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "epochs": epochs,
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "train_loss": train_result.training_loss,
        "total_train_time_sec": round(total_time_sec, 2),
        "peak_gpu_mem_mb": round(peak_mem_mb, 2)
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
