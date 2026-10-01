"""
predict.py
Runs inference using Base Model, LoRA adapter, or QLoRA adapter.
Parses output into structured JSON, captures latency, and saves predictions.
"""

import os
import re
import time
import json
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are the AI Triage Assistant for NeoBank.\n"
    "Your role is to triage customer support messages according to bank policies.\n"
    "You must output a single valid JSON object with the following fields:\n"
    "- \"category\": one of [\"card_payment\", \"card_declined\", \"transfer_delay\", \"refund_status\", \"account_access\", \"suspected_fraud\", \"general_question\"]\n"
    "- \"escalate\": boolean (true if the issue requires human agent intervention or account verification, false otherwise)\n"
    "- \"response\": a concise, policy-grounded response guiding the customer without requesting credentials or making unauthorized financial guarantees.\n\n"
    "Never ask for PIN, password, OTP, or CVV. Output JSON only."
)

def extract_json(raw_text: str):
    """Robustly extracts JSON dictionary from model generation."""
    raw_text = raw_text.strip()
    
    # Try finding markdown code block
    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_block:
        candidate = json_block.group(1)
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val, candidate
        except Exception:
            pass

    # Try finding any outer curly braces
    brace_match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val, candidate
        except Exception:
            pass

    # Direct json.loads if candidate is raw json dict
    try:
        val = json.loads(raw_text)
        if isinstance(val, dict):
            return val, raw_text
    except Exception:
        pass

    # Regex extraction fallback
    category_match = re.search(r'"category"\s*:\s*"([^"]+)"', raw_text)
    escalate_match = re.search(r'"escalate"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', raw_text)
    
    fallback = {
        "category": category_match.group(1) if category_match else "unknown",
        "escalate": (escalate_match.group(1).lower() == "true") if escalate_match else False,
        "response": response_match.group(1) if response_match else raw_text
    }
    return fallback, raw_text

def load_inference_pipeline(
    model_name: str,
    adapter_path: str = None,
    load_in_4bit: bool = False
):
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    compute_dtype = torch.float16
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )

    print(f"Loading base model from {model_name} (4bit={load_in_4bit})...")
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path and os.path.exists(adapter_path):
        print(f"Attaching LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model

def predict_single(
    message: str,
    tokenizer,
    model,
    max_new_tokens: int = 192,
    temperature: float = 0.1
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start

    # Decode only generated response
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    parsed_json, json_text = extract_json(raw_response)

    return {
        "raw_response": raw_response,
        "parsed": parsed_json,
        "latency_sec": round(elapsed, 4)
    }

def run_batch_prediction(
    input_file: str,
    output_file: str,
    tokenizer,
    model,
    limit: int = None
):
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    print(f"Running predictions on {len(records)} examples from {input_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    total_time = 0.0

    for idx, rec in enumerate(records):
        msg = rec["message"]
        pred = predict_single(msg, tokenizer, model)
        total_time += pred["latency_sec"]

        entry = {
            "id": rec.get("id", f"sample_{idx}"),
            "message": msg,
            "ground_truth_category": rec.get("category"),
            "ground_truth_escalate": rec.get("escalate"),
            "ground_truth_response": rec.get("response"),
            "predicted_category": pred["parsed"].get("category", "unknown"),
            "predicted_escalate": pred["parsed"].get("escalate", False),
            "predicted_response": pred["parsed"].get("response", ""),
            "raw_output": pred["raw_response"],
            "latency_sec": pred["latency_sec"]
        }
        results.append(entry)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(records):
            print(f"  Processed {idx + 1}/{len(records)} - Avg Latency: {total_time / (idx + 1):.3f}s")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Inference for Fintech Support Triage")
    parser.add_argument("--model_type", type=str, choices=["base", "lora", "qlora"], default="base")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--single_prompt", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_4bit = (args.model_type == "qlora")
    adapter = args.adapter_path
    if args.model_type == "lora" and not adapter:
        adapter = "outputs/checkpoints_lora"
    elif args.model_type == "qlora" and not adapter:
        adapter = "outputs/checkpoints_qlora"

    tokenizer, model = load_inference_pipeline(
        model_name=args.base_model,
        adapter_path=adapter,
        load_in_4bit=load_4bit
    )

    if args.single_prompt:
        res = predict_single(args.single_prompt, tokenizer, model)
        print("\n--- Model Response ---")
        print(json.dumps(res, indent=2))
    else:
        out_file = args.output_file or f"outputs/predictions/{args.model_type}_predictions.jsonl"
        run_batch_prediction(
            input_file=args.input_file,
            output_file=out_file,
            tokenizer=tokenizer,
            model=model,
            limit=args.limit
        )

if __name__ == "__main__":
    main()
