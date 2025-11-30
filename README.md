# PREHAB‑SVD

A unified research codebase implementing **Low‑Rank Prehab**, a lightweight pre‑conditioning stage that prepares neural networks for SVD-based compression by optimizing a Fisher‑aligned stable‑rank surrogate before truncation.  
This repository provides a full experimental pipeline for BERT, ViT, and (soon) LLaMA, modeling closely after modern SVD‑based compression frameworks while remaining modular, readable, and easy to extend.

---

## Overview

Low‑Rank Prehab inserts a brief pre‑conditioning phase *before* SVD truncation:

1. **Baseline fine‑tuning**  
2. **Activation whitening** (profiling Cholesky factors of layer inputs)
3. **Prehab stage** — stable‑rank / nuclear‑norm surrogate regularization in whitened space  
4. **SVD‑based compression sweeps** (raw and whitened‑space)  
5. **Optional LoRA recovery** (two‑stage U/V refinement—coming soon)  
6. **LLaMA support** (skeleton included; full implementation coming)

The goal is a consistent, comparable pipeline for understanding:
- raw‑space vs whitened‑space SVD,
- impact of stable‑rank pre-conditioning,  
- compression behavior across architectures.

---

## Directory Structure

```
models/
  bert/        # BERT-specific modules and folding utilities
  vit/         # ViT helpers, QKV splitting, legacy LoRA variants
  llama/       # Placeholder stubs for LLaMA-7B support
scripts/       # Main training, whitening, prehab, and SVD workflows
utils/         # Metrics, rank functions, data loaders, model ops
data/          # Dataset notes and pointers (no large files)
```

---

## Core Concepts

### 1. Whitening (Activation Standardization)
We estimate activation covariance matrices using several forward batches.  
A stabilized Cholesky or eigendecomposition yields matrices **L** such that:

```
X' = X L     # whitened activations
W' = W L     # whitened weights
```

Whitened space improves:
- conditioning of rank-surrogate gradients,
- interpretability of singular value decay,
- consistency of truncation across layers.

### 2. Low‑Rank Prehab
Prehab performs gradient updates in whitened weight space:

```
min      L_task(W X X^{-1})
    + λ * R_rank(W X)
```

Rank surrogate:
- **Stable rank** (BERT/LlaMA)
- **Nuclear norm surrogate** + **entropy effective rank** (ViT)

This gently shapes the spectrum toward compressible configurations *before* SVD.

### 3. Compression Modes

#### A. Raw SVD
Truncate singular values directly in `W`.

#### B. Whitened‑Space Thresholding
Apply truncation in whitened space (`W L`), then map back:

```
W ≈ (U_r Σ_r V_r^T) L^{-1}
```

Often yields better accuracy under high compression.

---

## Script Overview

All scripts use a unified interface:
```
--model {bert, vit, llama}
```

### **1. `scripts/finetune.py` — Baseline Pretraining / Fine‑tuning**
BERT:
```
python scripts/finetune.py   --model bert --glue_task sst2   --output_dir out/bert_ft
```

ViT:
```
python scripts/finetune.py   --model vit --imagenet_path /data/imagenet   --output_dir out/vit_ft
```

---

### **2. `scripts/whiten.py` — Activation Whitening Profiler**
Computes per-layer whitening matrices.

Example (BERT):
```
python scripts/whiten.py   --model bert --glue_task sst2   --input_model_path out/bert_ft/model.pth   --output_dir out/bert_whiten
```

Example (ViT):
```
python scripts/whiten.py   --model vit --imagenet_path /data/imagenet   --output_dir out/vit_whiten --prof_batches 50
```

---

### **3. `scripts/prehab.py` — Stable‑Rank Conditioning**
BERT:
```
python scripts/prehab.py   --model bert --glue_task sst2   --input_model_path out/bert_ft/model.pth   --input_whiten_path out/bert_whiten/whiten.pth   --output_dir out/bert_prehab --prehab_nbatch 600
```

ViT:
```
python scripts/prehab.py   --model vit --imagenet_path /data/imagenet   --output_dir out/vit_prehab   --epochs 2500 --prof_batches 50 --split_qkv
```

Produces:
- `prehab.pth` (BERT)
- `final_model_folded.pth` (ViT)

---

### **4. `scripts/svd.py` — Raw SVD Truncation Sweep**
```
python scripts/svd.py   --model bert --glue_task sst2   --input_model_path out/bert_prehab/prehab.pth   --output_csv out/bert_svd/results.csv
```

---

### **5. `scripts/whiten_thresholds.py` — Whitening-Space Truncation**
```
python scripts/whiten_thresholds.py   --model vit   --imagenet_path /data/imagenet   --whiten_path out/vit_whiten/whiten_vit.pth   --output_csv out/vit_wspace/results.csv --split_qkv
```

---

## Utilities

### `utils/metrics.py`
Includes:
- GLUE evaluators,
- ImageNet Top‑1,
- stable rank, entropy effective rank,
- spectrum diagnostics.

### `utils/model_utils.py`
- Folding/unfolding layers,
- whitened-space truncation helpers,
- seed control, misc wrapper utilities.

### `utils/data_utils.py`
- GLUE loaders,
- ImageNet dataloaders (via timm wrappers).

---

## Typical Workflow

1. **Finetune** the model on target task  
2. **Whiten** activations  
3. Run **Prehab** for low-rank conditioning  
4. Perform one of:
   - Raw SVD sweep
   - Whitening‑space truncation sweep  
5. (Optional) **LoRA refinement**  
6. Compare accuracy / perplexity vs compression

