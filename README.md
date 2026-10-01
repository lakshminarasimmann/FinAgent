# 💳 FinAgent: Fine-Tuning an LLM for Fintech Support Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-green.svg)](https://pytorch.org/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-orange.svg)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade fine-tuning project engineered for digital banking customer support triage. The system ingests inbound customer support queries and performs three core operations in a single structured inference pass:
1. **Issue Classification**: Accurately categorizes queries into one of 7 standardized banking support categories.
2. **Policy-Grounded Next Steps**: Generates clear, actionable guidance strictly aligned with documented digital bank operating procedures.
3. **Automated Escalation Routing**: Distinguishes self-service resolution paths from high-risk scenarios requiring human intervention (`escalate: true | false`).
4. **Zero-Trust Safety Guardrails**: Enforces non-negotiable security boundaries—never solicits credentials (PIN, password, OTP, CVV) and avoids unauthorized financial promises or outcome guarantees.

---

## 📑 Table of Contents
- [Executive Overview & Problem Statement](#-executive-overview--problem-statement)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Dataset Provenance & Data Engineering](#-dataset-provenance--data-engineering)
- [Digital Bank Policy Guide & Escalation Matrix](#-digital-bank-policy-guide--escalation-matrix)
- [Fine-Tuning Methodology & Implementation](#-fine-tuning-methodology--implementation)
- [Empirical Evaluation & Measured Results](#-empirical-evaluation--measured-results)
- [Hardware & Resource Optimization Insights](#-hardware--resource-optimization-insights)
- [Quickstart & Usage Guide](#-quickstart--usage-guide)
- [Repository Structure](#-repository-structure)

---

## 🎯 Executive Overview & Problem Statement

Digital banks and fintechs receive thousands of inbound support requests daily regarding failed transfers, unrecognized card charges, delayed refunds, and login blocks. Off-the-shelf general-purpose LLMs struggle in this domain:
- **Schema Failure**: Raw LLMs generate conversational prose rather than the rigid JSON required for CRM routing engines.
- **Dangerous Escalation Gaps**: Base models frequently attempt to "resolve" suspected fraud or interbank transfer failures via conversational chat instead of urgently escalating to specialized fraud and operations teams.
- **Security Vulnerabilities**: Without targeted fine-tuning, models may accept or solicit sensitive customer credentials.

**Project Core Question**: *Does Parameter-Efficient Fine-Tuning (PEFT) using LoRA or QLoRA enable a small open-weight LLM to achieve enterprise-grade accuracy, policy adherence, and safety guardrails on consumer hardware (e.g., 6GB VRAM)?*

---

## 🏗️ System Architecture & Pipeline

```
Public Banking Data Sources               Digital Bank Policy Guide
[PolyAI/banking77 + Bitext]              [policies/support_policy.md]
            │                                         │
            ▼                                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Data Engineering & Redaction Pipeline (prepare_data.py)      │
 │  - Intent-to-category mapping                               │
 │  - Regex-based PII & sensitive credential redaction         │
 │  - Deduplication & text normalization                       │
 │  - Scenario-based Train (70%) / Val (15%) / Test (15%)      │
 │  - 10-Case Adversarial Regression Suite                     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Chat Template Formatting & Tokenization                     │
 │ Base Model: Qwen/Qwen2.5-0.5B-Instruct                      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
 ┌───────────────────────────┐     ┌───────────────────────────┐
 │ FP16 LoRA Fine-Tuning     │     │ 4-Bit QLoRA Fine-Tuning   │
 │ - Rank 16, Alpha 32       │     │ - NF4 Double Quantization │
 │ - All linear projections  │     │ - Peak VRAM: ~3.4 GB      │
 └─────────────┬─────────────┘     └─────────────┬─────────────┘
               └────────────────┬────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Comparative Evaluation Framework (evaluate.py)              │
 │  - JSON Schema Validity %                                   │
 │  - Category Classification Accuracy & Macro F1              │
 │  - Escalation Precision, Recall & Missed Escalation Rate    │
 │  - Policy Grounding & Safety Guardrail Adherence            │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Interactive Streamlit Demo & Triage Console (app.py)        │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Provenance & Data Engineering

To ensure realism and compliance with data governance standards, no real customer bank accounts or private data were utilized:

### 1. Data Sources
- **`PolyAI/banking77`**: 13,083 authentic English customer queries across 77 fine-grained commercial banking intents (card delivery, failed transactions, exchange rates, compromised cards).
- **`bitext/Bitext-customer-support-llm-chatbot-training-dataset`**: 26,872 retail customer service dialogues covering payment disputes, refund inquiries, and account recovery.

### 2. Automated PII & Credential Sanitization
The data processing pipeline ([src/prepare_data.py](file:///c:/Users/PRASANNA/OneDrive/Desktop/FinAgent/src/prepare_data.py)) scans every text sample using automated regular expressions to eliminate or mask personal data:
- **Card Numbers (13–16 digits)** $\rightarrow$ `[REDACTED_CARD]`
- **Email Addresses** $\rightarrow$ `[REDACTED_EMAIL]`
- **Phone Numbers** $\rightarrow$ `[REDACTED_PHONE]`
- **Social Security Numbers** $\rightarrow$ `[REDACTED_SSN]`
- **Passcodes & PINs** $\rightarrow$ `[REDACTED_CODE]`

### 3. Stratified Dataset Distribution
All duplicates were eliminated, and data was split by category to prevent data leakage:
- **Training Set**: 868 samples (balanced across all 7 categories)
- **Validation Set**: 186 samples
- **Held-Out Test Set**: 186 samples (50 used for standardized benchmarking)
- **Adversarial Regression Suite**: 10 curated stress tests ([data/regression/regression_test.jsonl](file:///c:/Users/PRASANNA/OneDrive/Desktop/FinAgent/data/regression/regression_test.jsonl)), including social engineering phone scams, prompt injection overrides, and aggressive refund demands.

---

## 📖 Digital Bank Policy Guide & Escalation Matrix

All responses are strictly grounded in [policies/support_policy.md](file:///c:/Users/PRASANNA/OneDrive/Desktop/FinAgent/policies/support_policy.md):

| Category | Description | Policy Prescribed Next Steps | Mandatory Escalation? |
| :--- | :--- | :--- | :---: |
| **`card_payment`** | Unrecognized charge or billing discrepancy | Freeze card in-app; check pending status; submit in-app dispute form if settled. | `true` |
| **`card_declined`** | In-store, online, or ATM card failure | Check spending limits and toggle; unblock PIN in app via biometric authentication. | `false` (Unless swallowed by ATM) |
| **`transfer_delay`** | Inbound or outbound bank transfer delay | Verify routing info; explain 1–3 day domestic or 3–5 day wire processing windows. | `true` if window exceeded |
| **`refund_status`** | Merchant return/reversal not appearing | Clarify 5–10 business day settlement window; obtain merchant ARN reference. | `false` (Unless >10 business days) |
| **`account_access`** | Passcode reset, locked account, biometric | Direct to self-service "Forgot Passcode" flow; verify registered device. | `true` if phone lost/2FA reset |
| **`suspected_fraud`** | Stolen card/device, phishing scam, unauthorized | Urgent card freeze; reset credentials; immediate routing to Fraud Specialist Team. | **`true` (Mandatory)** |
| **`general_question`**| Fee schedules, limits, app capabilities | Reference Help Center; explain fee-free ATM tier and domestic features. | `false` |

---

## ⚙️ Fine-Tuning Methodology & Implementation

### Model Selection
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Rationale**: Exceptionally strong multilingual and structured JSON capabilities in a compact ~500M parameter footprint, permitting fast iteration and low memory usage.

### Target Output Schema
The model is fine-tuned to emit strictly valid JSON:
```json
{
  "category": "suspected_fraud",
  "escalate": true,
  "response": "URGENT: Mudra Digital Bank will never call asking for your one-time verification passcode (OTP), MPIN, or password. Freeze your card immediately in the app and change your password. We are escalating this case directly to our Fraud Prevention Team."
}
```

### Hyperparameters & LoRA Adapter Configuration
- **LoRA Rank ($r$)**: 16
- **LoRA Alpha ($\alpha$)**: 32
- **LoRA Dropout**: 0.05
- **Target Modules**: All linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Learning Rate**: $2.0 \times 10^{-4}$ with Cosine Learning Rate Schedule
- **Warmup Ratio**: 0.05
- **Optimizer**: AdamW with Weight Decay 0.01
- **Effective Batch Size**: 16 (configured as `per_device_train_batch_size: 2` with `gradient_accumulation_steps: 8` for optimal GPU memory headroom)

---

## 📈 Empirical Evaluation & Measured Results

All models were evaluated on the **exact same held-out test split** ($N=50$) using the quantitative evaluation suite ([src/evaluate.py](file:///c:/Users/PRASANNA/OneDrive/Desktop/FinAgent/src/evaluate.py)).

### Complete Comparative Benchmark Table

| Metric | Base Model (`Qwen2.5-0.5B`) | LoRA (3 Epochs) | **LoRA (5 Epochs)** | **QLoRA (5 Epochs, 4-Bit NF4)** |
| :--- | :---: | :---: | :---: | :---: |
| **Valid Structured JSON %** | 34.0% | **100.0%** | **100.0%** | **100.0%** |
| **Category Classification Accuracy** | 12.0% | **94.0%** | **94.0%** | **94.0%** |
| **Category Macro F1** | 12.8% | **92.8%** | **93.2%** | **92.4%** |
| **Escalation Recall (Capturing Fraud/Delays)** | 35.3% | **94.1%** | **94.1%** | **94.1%** |
| **Missed Escalation Rate (Lower is safer)** | 64.7% | **5.9%** | **5.9%** | **5.9%** |
| **Policy Next-Step Grounding Score** | 16.0% | 88.0% | **90.0%** | **90.0%** |
| **Credential Solicitation Violations** | 0 | **0** | **0** | **0** |
| **Safety & Guardrail Compliance** | 100.0% | **100.0%** | **100.0%** | **100.0%** |
| **Average Safety Disclaimer Inclusion** | 0.0% | 12.0% | **14.0%** | **14.0%** |
| **Final Training Loss** | N/A | 0.3663 | **0.2853** | **0.3102** |
| **Best Evaluation Loss (Epoch 3)** | N/A | 0.1478 | **0.1444** | **0.1462** |
| **Total Training Time** | Baseline | 7.29 mins (437s) | **16.14 mins (968s)** | **46.96 mins (2,817s)** |
| **Peak GPU VRAM Allocated** | ~1.2 GB (Inference) | 5,685 MB | **3,416 MB (-40.0%)** | **2,308 MB (-59.4%)** |

---

## 🔬 Hardware & Resource Optimization Insights

### 1. Training Convergence & Overfitting Analysis (LoRA vs QLoRA)
Tracking evaluation loss across epochs revealed the exact convergence curve for both models on this dataset:

| Epoch | LoRA (FP16) Eval Loss | QLoRA (4-Bit NF4) Eval Loss | Model Status & Phase |
| :---: | :---: | :---: | :--- |
| **Epoch 1** | `0.1631` | `0.1629` | Rapid learning of syntax & JSON structure |
| **Epoch 2** | `0.1473` | `0.1499` | Aligning policy response rules |
| **Epoch 3** | **`0.1444`** | **`0.1462`** | 🎯 **Optimal Validation Convergence** |
| **Epoch 4** | `0.1459` | `0.1479` | Loss begins to plateau |
| **Epoch 5** | `0.1491` | `0.1511` | Overfitting begins (validation loss creeps up) |

> **Key Takeaway**: Both FP16 LoRA and 4-bit QLoRA converge to their minimum validation loss at **Epoch 3**, with nearly identical quality (`0.1444` vs `0.1462`). Training beyond 5 epochs produces diminishing returns and increases overfitting risk.

### 2. GPU Memory Footprint & Compute Trade-Offs (6GB RTX 4050)
- **Initial LoRA (Batch Size 4, Grad Acc 4)**: Consumed **5,685 MB VRAM** (93% capacity of the 6GB card). Under Windows WDDM, desktop memory fragmentation caused occasional cuBLAS driver stalls.
- **Optimized LoRA (Batch Size 2, Grad Acc 8)**: Consumed only **3,416 MB VRAM** (**40% memory reduction**), completely eliminating memory collisions while preserving the exact same effective batch size of 16. Total training runtime: **16.14 minutes**.
- **QLoRA (4-bit NF4)**: Consumed a mere **2,308 MB VRAM** (**59.4% memory reduction**). This leaves over 3.8 GB of VRAM free, making it completely stable on low-memory or shared GPUs (e.g., 4GB laptop GPUs). Because 4-bit dequantization/quantization occurs dynamically at every forward and backward pass, training takes **46.96 minutes** (~2.9× slower than LoRA, but with unmatched memory efficiency).

---

## 🚀 Quickstart & Usage Guide

### 1. Setup & Environment
Ensure Python 3.10+ and CUDA-enabled PyTorch are installed:
```powershell
pip install torch transformers peft accelerate bitsandbytes datasets streamlit pyyaml scikit-learn
```

### 2. Download Datasets
Downloads `PolyAI/banking77` and `Bitext` into `data/raw/`:
```powershell
python src\download_datasets.py
```

### 3. Clean, Redact, and Prepare Data
Generates train, validation, test splits, and regression tests in `data/processed/`:
```powershell
python src\prepare_data.py
```

### 4. Fine-Tuning
- **Run LoRA Fine-Tuning (5 Epochs)**:
  ```powershell
  python src\train_sft.py --config configs\lora.yaml --epochs 5 --output_dir outputs\checkpoints_lora_5ep
  ```
- **Run QLoRA Fine-Tuning (4-Bit NF4)**:
  ```powershell
  python src\train_sft.py --config configs\qlora.yaml --epochs 5 --output_dir outputs\checkpoints_qlora_5ep
  ```

### 5. Run Evaluations
Score model predictions across all benchmarks:
```powershell
python src\evaluate.py
```

### 6. Launch the Interactive Web Demo
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore live customer support message triage, side-by-side model outputs, and regression test suites.

---

## 📁 Repository Structure

```
FinAgent/
├── data/
│   ├── raw/
│   │   ├── banking77/       # Raw PolyAI/banking77 train & test splits
│   │   └── bitext/          # Raw Bitext customer support dataset
│   ├── processed/
│   │   ├── train.jsonl      # 868 sanitized & balanced training samples
│   │   ├── val.jsonl        # 186 validation samples
│   │   └── test.jsonl       # 186 held-out test samples
│   └── regression/
│       └── regression_test.jsonl  # 10 curated adversarial & edge cases
├── policies/
│   └── support_policy.md    # NeoBank customer support policy & triage rules
├── configs/
│   ├── lora.yaml            # FP16 LoRA configuration (batch size 2, grad acc 8)
│   └── qlora.yaml           # 4-bit NF4 QLoRA configuration
├── src/
│   ├── download_datasets.py # Automated HF dataset downloader
│   ├── prepare_data.py      # Sanitization, redaction, deduping, and splitting
│   ├── train_sft.py         # SFT training loop with memory & time profiling
│   ├── predict.py           # Robust JSON extraction & inference script
│   └── evaluate.py          # Metric scoring (Accuracy, F1, Recall, Safety)
├── outputs/
│   ├── checkpoints_lora_5ep/# Trained 5-epoch LoRA adapter & tokenizer
│   ├── checkpoints_qlora_5ep/# Trained 5-epoch 4-bit QLoRA adapter & tokenizer
│   ├── predictions/         # Model predictions on held-out test data
│   └── metrics/             # JSON evaluation benchmarks & training logs
├── app.py                   # Interactive Streamlit triage web application
└── README.md                # Comprehensive project documentation
```

---

## 🛡️ License & Ethics
This project uses public datasets (`PolyAI/banking77` and `Bitext`) under appropriate research and educational licensing. All personal identifiers, simulated account numbers, and credentials have been redacted. The triage assistant is strictly designed as an informational routing tool and does not perform autonomous financial transactions.
