# PREHAB-SVD Unified Repository

This repository unifies previously separate BERT (GLUE) and ViT (ImageNet) codebases under a common workflow for:

1. Baseline finetuning
2. Activation whitening (profiling Cholesky factors of layer inputs)
3. PREHAB: low-rank regularization in the whitened space (effective rank / nuclear norm surrogate)
4. SVD-based compression sweeps in raw space
5. Whitened-space threshold (rank) sweeps
6. (Planned) LoRA two-stage refinement after compression
7. (Planned) Llama support (all stubs currently)

The goal is a consistent experimental pipeline comparing raw-space versus whitened-space compression and the effect of PREHAB rank regularization before SVD.

---
## Directory Structure
```
models/
  bert/        # Original BERT-specific logic (to be progressively migrated)
  vit/         # ViT helpers & legacy LoRA SVD variants
  llama/       # Placeholder stubs
scripts/       # Unified executable workflows (entry points)
utils/         # New shared utilities (metrics, data, model ops)
data/          # Dataset notes / pointers (no large files stored)
```

---
## Core Concepts

### Whitening
We profile layer input activations over several batches to form empirical covariance matrices. A robust Cholesky (or stabilized eigendecomposition) yields lower-triangular matrices L such that X L approximately standardizes activations. Transforming weights W → W L places optimization into a decorrelated space ("whitened space") where effective rank constraints are cleaner and better conditioned.

### PREHAB
PREHAB optimizes additive deltas in whitened weight space with a rank surrogate penalty:
- BERT: stable (effective) rank via differentiable spectral norm approximation.
- ViT: nuclear norm surrogate + entropy-based diagnostic effective rank.
It produces a model whose weights tend toward lower effective rank before explicit truncation.

### Compression Workflows
A. PREHAB → SVD: Finetune → Whiten → PREHAB → SVD sweep (raw) or whitened threshold truncation.
B. Whiten → SVD (+ optional future LoRA): Finetune → Whiten → compress directly (raw vs whitened) → optional LoRA refinement.

### Raw vs Whitened SVD
Raw SVD applies rank truncation directly to W. Whitened thresholding performs truncation in W L space, then maps back, often yielding better parameter efficiency due to decorrelated dimension scaling.

---
## Scripts Overview

All scripts accept a common flag `--model {bert, vit, llama}`. Llama is currently a placeholder.

### 1. Finetuning: `scripts/finetune.py`
- BERT: GLUE task finetuning using HuggingFace transformers.
- ViT: ImageNet finetuning / linear probe using timm pretrained backbone.
- Outputs: state dict at `<output_dir>/model.pth` plus config JSON.

Example (BERT SST-2):
```
python scripts/finetune.py \
  --model bert --glue_task sst2 \
  --output_dir out/bert_ft
```
Example (ViT ImageNet):
```
python scripts/finetune.py \
  --model vit --imagenet_path /data/imagenet \
  --output_dir out/vit_ft
```

### 2. Whitening Profiler: `scripts/whiten.py`
Generates whitening matrices (Cholesky factors) per targeted layer.
- BERT: list of L matrices (`whiten.pth`)
- ViT: dict of lists (`whiten_vit.pth` for qkv, proj, fc1, fc2, head)

Example:
```
python scripts/whiten.py --model bert --glue_task sst2 \
  --input_model_path out/bert_ft/model.pth \
  --output_dir out/bert_whiten
```
```
python scripts/whiten.py --model vit --imagenet_path /data/imagenet \
  --output_dir out/vit_whiten --prof_batches 50
```

### 3. PREHAB Rank Regularization: `scripts/prehab.py`
Optimizes in whitened space with rank surrogate penalty.
- BERT: stable rank surrogate (power iteration spectral norm)
- ViT: nuclear norm surrogate + entropy effective rank
- Produces folded model state at `<output_dir>/prehab.pth` (BERT) or `final_model_folded.pth` (ViT) plus logs & metrics.

Example:
```
python scripts/prehab.py --model bert --glue_task sst2 \
  --input_model_path out/bert_ft/model.pth \
  --input_whiten_path out/bert_whiten/whiten.pth \
  --output_dir out/bert_prehab --prehab_nbatch 600
```
```
python scripts/prehab.py --model vit --imagenet_path /data/imagenet \
  --output_dir out/vit_prehab --epochs 2500 --prof_batches 50 --split_qkv
```

### 4. Raw SVD Sweep: `scripts/svd.py`
Applies SVD rank truncations directly to selected modules.
- Records accuracy / metric vs retained rank fraction.
- ViT supports optional QKV splitting.

Example:
```
python scripts/svd.py --model bert --glue_task sst2 \
  --input_model_path out/bert_prehab/prehab.pth \
  --output_csv out/bert_svd/results.csv
```
```
python scripts/svd.py --model vit --imagenet_path /data/imagenet \
  --output_csv out/vit_svd/results.csv --split_qkv
```

### 5. Whitened Threshold Sweep: `scripts/whiten_thresholds.py`
Performs truncation in whitened space (W L). Maps result back to raw weight domain.
- Outputs CSV with threshold, accuracy/top1, params kept percentage.

Example:
```
python scripts/whiten_thresholds.py --model bert --glue_task sst2 \
  --input_model_path out/bert_prehab/prehab.pth \
  --whiten_path out/bert_whiten/whiten.pth \
  --output_csv out/bert_wspace/results.csv
```
```
python scripts/whiten_thresholds.py --model vit --imagenet_path /data/imagenet \
  --whiten_path out/vit_whiten/whiten_vit.pth \
  --output_csv out/vit_wspace/results.csv --split_qkv
```

---
## Utilities

### `utils/metrics.py`
Shared evaluation & rank metrics:
- `eval_glue_model` (classification/regression)
- `compute_top1` ImageNet accuracy
- `stable_rank`, `overall_effective_rank`, `entropy_effective_rank`

### `utils/model_utils.py`
Model operations:
- `set_seed`, BERT module referencing
- Whitening-space truncation helpers (`truncate_wspace`, `truncate_wspace_modules`)
- Delegated folding for BERT wrappers

### `utils/data_utils.py`
Data loading:
- GLUE dataloader with tokenizer abstraction
- ImageNet loader delegation to `models/vit/imagenet.py`

---
## Workflow Summary

1. Finetune baseline (establish task accuracy / top-1)
2. Profile whitening matrices (captures activation correlations)
3. PREHAB (rank regularization in decorrelated space)
4. Evaluate compression strategies:
   - Raw SVD sweep
   - Whitened threshold sweep
5. (Planned) Apply LoRA refinement after truncation
6. Compare metrics (accuracy vs parameter retention vs effective rank ratios)

---
## Experimental Rationale

- Whitening improves conditioning of rank penalties: decorrelated dimensions avoid biasing toward directions with larger variance.
- PREHAB before compression can lower intrinsic rank, yielding better performance at aggressive truncation ratios.
- Whitened-space truncation preserves directions aligned with activation covariance, often outperforming naive SVD on raw weights.
- Entropy-based effective rank provides a scale-sensitive diagnostic complementary to stable rank objectives.

---
## Roadmap (Pending Items)

- Migrate all remaining rank/eval helpers from `models/bert/utils.py` into `utils/`.
- Centralize robust SVD / Cholesky numerics (currently duplicated in scripts).
- Integrate unified LoRA refinement (`--lora` flag) across workflows.
- Implement full Llama support (data loading, whitening, PREHAB).
- Add validation for whitening matrix/module length mismatches.
- Package as installable module (`setup.py` / `pyproject.toml`, `requirements.txt`).
- Add orchestration (Makefile or bash pipelines) for full experiment runs.
- Expand metrics JSON schema for comparative plots.

---
## Environment & Dependencies
Minimal requirements (to be formalized):
```
torch
transformers
datasets
evaluate
timm
numpy
```
Install example:
```
pip install torch transformers datasets evaluate timm numpy
```

---
## Citation / Reference
If you use or extend this repository, please cite the PREHAB method (placeholder for eventual paper / preprint).

---
## License
Specify license here (MIT / Apache-2.0 etc.).

---
## Contact
Open issues for bugs or feature requests. Contributions welcome for Llama integration & LoRA unification.
