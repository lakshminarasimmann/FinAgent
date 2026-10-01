"""
Script to download and inspect BANKING77 and Bitext Customer Support datasets.
Saves local raw caches/copies into data/raw/.
"""

import os
import json
from datasets import load_dataset

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(os.path.join(RAW_DIR, "banking77"), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, "bitext"), exist_ok=True)

def download_banking77():
    print(">>> [1/2] Downloading PolyAI/banking77...")
    try:
        ds = load_dataset("PolyAI/banking77")
    except Exception as e:
        print(f"Failed with PolyAI/banking77, trying banking77: {e}")
        ds = load_dataset("banking77")

    print(f"BANKING77 splits: {list(ds.keys())}")
    for split_name in ds.keys():
        print(f"  {split_name}: {len(ds[split_name])} rows")
        sample = ds[split_name][0]
        print(f"  Sample {split_name}[0]: {sample}")

    # Extract label names
    features = ds["train"].features
    label_names = features["label"].names if "label" in features and hasattr(features["label"], "names") else None
    print(f"  Total intent classes: {len(label_names) if label_names else 'N/A'}")

    # Save to disk as jsonlines for reproducibility
    train_path = os.path.join(RAW_DIR, "banking77", "train.jsonl")
    test_path = os.path.join(RAW_DIR, "banking77", "test.jsonl")
    labels_path = os.path.join(RAW_DIR, "banking77", "label_names.json")

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    with open(train_path, "w", encoding="utf-8") as f:
        for row in ds["train"]:
            f.write(json.dumps({"text": row["text"], "label": row["label"], "intent": label_names[row["label"]] if label_names else row["label"]}) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for row in ds["test"]:
            f.write(json.dumps({"text": row["text"], "label": row["label"], "intent": label_names[row["label"]] if label_names else row["label"]}) + "\n")

    print(f"BANKING77 saved to {os.path.join(RAW_DIR, 'banking77')}")

def download_bitext():
    print("\n>>> [2/2] Downloading bitext/Bitext-customer-support-llm-chatbot-training-dataset...")
    try:
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    except Exception as e:
        print(f"Failed loading Bitext dataset: {e}")
        return

    print(f"Bitext splits: {list(ds.keys())}")
    for split_name in ds.keys():
        print(f"  {split_name}: {len(ds[split_name])} rows")
        sample = ds[split_name][0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample {split_name}[0]: {sample}")

    # Save a clean copy
    for split_name in ds.keys():
        out_path = os.path.join(RAW_DIR, "bitext", f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds[split_name]:
                f.write(json.dumps(dict(row)) + "\n")

    print(f"Bitext dataset saved to {os.path.join(RAW_DIR, 'bitext')}")

if __name__ == "__main__":
    download_banking77()
    download_bitext()
    print("\nAll dataset downloads and raw staging complete!")
