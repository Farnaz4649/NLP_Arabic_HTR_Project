# Arabic Handwritten Text Recognition via Knowledge Distillation

**MAI 656 — Natural Language Processing | Canadian University Dubai | Spring 2026**

A knowledge-distillation pipeline that transfers Arabic handwriting recognition capability from GPT-4.1-mini (teacher) to a locally deployable Qwen2.5-VL-7B + LoRA adapter (student). The fine-tuned student achieves **CER 0.3283** on 280 held-out images, a **29.2% improvement** over the zero-shot baseline and outperforming off-the-shelf EasyOCR.

---

## Pipeline Overview

```
AHTD Images (1,400)
       │
       ▼
┌─────────────────┐
│  GPT-4.1-mini   │  ← Teacher labeling (JSON output with confidence)
│    (Teacher)    │
└────────┬────────┘
         │  1,400 labeled samples (JSONL)
         ▼
┌─────────────────┐
│  Data Transform │  ← Convert GPT format → Qwen conversational format
│  Train/Eval     │     80/20 split: 1,120 train / 280 eval
│  Split          │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Train      Eval
(1,120)    (280)
    │
    ▼
┌─────────────────────────────┐
│  Qwen2.5-VL-7B-Instruct     │
│  4-bit NF4 quantization     │  ← Base model (~8 GB VRAM)
│  + LoRA (r=32, α=64)        │     3 epochs, 1,680 steps, A100
│  checkpoint-1120 (epoch 2)  │     Wall-clock: 1h 30min
└────────────┬────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│              Evaluation (280 samples)           │
│                                                 │
│  CER: 0.3283    WER: 0.6554                    │
│  Exact match: 1.07%   Valid JSON: 92.14%        │
│  Avg inference: 3.83 s/sample                   │
└────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│         Downstream NLP (CAMeL Tools)            │
│  NER: 122 entities (LOC 55, PERS 31, ORG 12)   │
│  POS: noun 33.9%, verb 14.2%, prep 11.7%        │
└─────────────────────────────────────────────────┘
```

---

## Results Summary

### Baseline Comparison (280-sample eval set, all vs GPT pseudo-labels)

| Model | CER | WER | Valid JSON |
|---|---|---|---|
| **Qwen2.5-VL-7B + LoRA (checkpoint-1120)** | **0.3283** | **0.6554** | 92.14% |
| Qwen2.5-VL-7B zero-shot | 0.4639 | 0.7755 | 97.50% |
| EasyOCR (Arabic, off-the-shelf) | 0.4764 | 1.0582 | N/A |
| TrOCR (microsoft/trocr-base-handwritten) | ❌ Discarded | | English-only tokenizer |

Fine-tuning improvement over zero-shot: **−29.2% CER**, **−15.5% WER**

### Normalization Impact

Applying 5 standard Arabic normalization steps (alef, taa marbuta, diacritics, kashida, punctuation) to both reference and hypothesis before scoring:

| Model | Raw CER | Norm CER | CER ↓% | Raw WER | Norm WER | WER ↓% |
|---|---|---|---|---|---|---|
| Qwen + LoRA | 0.3283 | 0.3043 | 7.3% | 0.6554 | 0.6160 | 6.0% |
| Qwen zero-shot | 0.4442 | 0.3506 | 21.1% | 0.7549 | 0.6931 | 8.2% |
| EasyOCR (10 samples) | 0.4377 | 0.4007 | 8.5% | 1.0388 | 0.9608 | 7.5% |

The large zero-shot normalization benefit (21.1%) reveals that much of its apparent error is orthographic convention mismatch rather than genuine misreading.

### Training Curve

| Epoch | Step | Eval Loss |
|---|---|---|
| 1 | 560 | 4.8740 |
| **2 (best)** | **1120** | **4.8603** |
| 3 | 1680 | 4.8669 |

Mild overfitting at epoch 3; `checkpoint-1120` selected for all evaluation.

---

## Repository Structure

```
NLP_Arabic_HTR_Project/
│
├── notebooks/                      # One notebook per pipeline stage
│   ├── NB_01_setup.ipynb           # Environment setup
│   ├── NB_02_data_preprocessing.ipynb  # Image validation and inspection
│   ├── NB_03_data_collection.ipynb     # GPT teacher labeling (API calls)
│   ├── NB_04_data_transformation.ipynb # GPT → Qwen format conversion
│   ├── NB_05_finetuning.ipynb      # LoRA fine-tuning (requires A100)
│   ├── NB_06_evaluation.ipynb      # CER/WER evaluation
│   ├── NB_07_downstream_nlp.ipynb  # NER + POS tagging (CAMeL Tools)
│   ├── NB_08_ocr_baseline.ipynb    # EasyOCR baseline
│   ├── NB_09_zero-Shot_qwen_baseline.ipynb  # Zero-shot Qwen baseline
│   ├── NB_10_normalization.ipynb   # Arabic text normalization impact
│   └── NB_11_perplexity.ipynb      # N-gram LM perplexity analysis
│
├── data/
│   ├── train/train.jsonl           # 1,120-sample training split
│   └── eval/eval.jsonl             # 280-sample evaluation split
│                                   # (images stored as base64 inside JSONL)
│
├── logs/
│   ├── run-1/
│   │   ├── eval_results_checkpoint-1120.json  # Primary eval results
│   │   ├── eval_results_zero_shot.json         # Zero-shot baseline
│   │   ├── training_log.json                   # Full training history
│   │   └── training_curve.png                  # Loss curve visualization
│   ├── stage6/
│   │   ├── ner_results.json        # NER comparison (GPT vs Qwen)
│   │   └── pos_results.json        # POS tagging results
│   ├── stage7/
│   │   └── easyocr_results.json    # EasyOCR baseline results
│   ├── stage10/
│   │   └── normalization_results.json  # Before/after normalization metrics
│   └── stage11/
│       └── perplexity_results.json     # N-gram LM perplexity scores
│
└── camel_data/                     # CAMeL Tools model data (NER + morphology)
    ├── catalogue.json
    ├── versions.json
    └── data/
        ├── ner/arabert/            # AraBERT NER model files
        ├── morphology_db/calima-msa-r13/   # CALIMA morphology DB
        └── disambig_mle/calima-msa-r13/    # MLE disambiguator
```

> **Note:** The `models/` directory (LoRA adapter checkpoints, ~1 GB each) is excluded from this repository via `.gitignore`. The adapter can be reproduced by running `NB_05_finetuning.ipynb` on an A100 GPU with the data files provided here.

---

## Setup and Reproduction

### Prerequisites

- Google Colab Pro (A100 GPU for NB_05, NB_06, NB_09; CPU/T4 for all others)
- Google Drive with sufficient storage (~5 GB for adapter checkpoints)
- OpenAI API key (NB_03 only, for teacher labeling)

### Quick Start

**1. Mount Drive and set project root** (all notebooks):
```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = '/content/drive/MyDrive/CUD files/NLP/NLP_Arabic_HTR_Project'
```

**2. Run notebooks in order:**

| Notebook | GPU Required | Purpose |
|---|---|---|
| NB_01 → NB_04 | No | Setup, data collection, transformation |
| NB_05 | A100 | LoRA fine-tuning (~90 min) |
| NB_06 | A100 | Evaluation (~45 min) |
| NB_07 | No | NER + POS with CAMeL Tools |
| NB_08 | No (T4 faster) | EasyOCR baseline |
| NB_09 | A100 | Zero-shot Qwen baseline (~45 min) |
| NB_10 | No | Normalization analysis |
| NB_11 | No | Perplexity analysis |

**3. If continuing from checkpoint** (skipping NB_01–04):
- Data files `train.jsonl` and `eval.jsonl` are already in this repo under `data/`
- Start directly from NB_05

### Key Dependencies

```python
# Fine-tuning (NB_05, NB_06, NB_09)
transformers>=4.40
peft>=0.10
bitsandbytes>=0.43
accelerate>=0.30
qwen-vl-utils
jiwer

# Downstream NLP (NB_07) — install without kernel restart, no version pinning
camel-tools --no-deps
cachetools==5.5.0
transformers>=4.0,<4.44.0   # camel-tools requires old transformers API

# Baseline (NB_08)
easyocr
jiwer

# Analysis (NB_10, NB_11)
jiwer
nltk
```

> **camel-tools path note:** CAMeL Tools requires a path with no spaces for its data directory. Symlink or copy `camel_data/` to `/content/camel_data` at session start and set `os.environ['CAMELTOOLS_DATA'] = '/content/camel_data'` before importing.

---

## Model Configuration

### LoRA Adapter (NB_05)

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| Quantization | 4-bit NF4, double quant, bf16 compute |
| LoRA rank (r) | 32 |
| LoRA alpha (α) | 64 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down proj |
| Trainable params | 95,178,752 (1.13% of 8.39B total) |
| Max sequence length | 8,192 tokens |
| Epochs | 3 |
| Learning rate | 2e-5 (cosine, 10% warmup) |
| Optimizer | AdamW fused |
| Hardware | NVIDIA A100-SXM4-40GB |
| Wall-clock time | 1h 30min |

---

## Dataset

**Arabic Handwritten Text Dataset (AHTD)**
- 1,400 images selected, split 80/20: 1,120 train / 280 eval
- Images stored as base64-encoded PNG inside JSONL (no separate image files needed)
- Teacher labels: GPT-4.1-mini, output format `{"transcription": "...", "confidence": "high|medium|low"}`
- Evaluation reference: GPT pseudo-labels (human ground truth not integrated)

---

## Known Limitations

- All CER/WER metrics are computed against GPT pseudo-labels, not human ground truth. AHTD has human annotations but they were not aligned to the pipeline's image ordering.
- `checkpoint-1120` is the best checkpoint but the adapter weights are not included in this repo due to size. Re-run NB_05 to reproduce them.
- CAMeL Tools requires a no-space path for its data directory. A symlink workaround is documented in NB_07.
- EasyOCR results in NB_10 and NB_11 are based on 10 stored samples only; the full 280-sample inference must be re-run to get complete normalization/perplexity figures for EasyOCR.

---

## Citation

If you use this pipeline or results, please cite:

```
@misc{arabic_htr_distillation_2026,
  title   = {Arabic Handwritten Text Recognition via Knowledge Distillation},
  author  = {Azarparand, Farnaz},
  year    = {2026},
  note    = {MAI 656 NLP Course Project, Canadian University Dubai}
}
```

---

## Course Information

**Course:** MAI 656 — Natural Language Processing  
**Institution:** Canadian University Dubai  
**Term:** Spring 2026  
**Supervisor:** Dr. Arash Kermani
