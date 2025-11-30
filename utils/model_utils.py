# Utility functions for model operations
# Move set_seed, count_params, get_module_references, truncate_wspace_modules here

"""Shared model utility functions.
Will progressively replace duplicated logic in models/bert/utils.py and ViT scripts.

Functions:
  set_seed(seed)
  get_bert_module_references(model, q=True, k=True, v=True, attn_out=True, fc1=True, fc2=True)
  count_params(modules)
  truncate_wspace(W, L, keep_ratio)
  truncate_wspace_modules(modules, L_list, keep_ratio, device)
  fold_in_deltas_bert(model, device, to_copy=False)  (thin wrapper delegating to legacy utils if present)

Note: For now we import models.bert.utils for backward compatibility.
"""
from __future__ import annotations
import math
import copy
import torch
from typing import List, Tuple

try:
    from models.bert import utils as bert_utils_legacy  # optional
except Exception:
    bert_utils_legacy = None

# --------------- Seed -----------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

# --------------- BERT module references -----------------
def get_bert_module_references(model, q=True, k=True, v=True, attn_out=True, fc1=True, fc2=True) -> List[Tuple[object, str, torch.nn.Module]]:
    refs = []
    for block in model.bert.encoder.layer:
        if q: refs.append((block.attention.self, 'query', block.attention.self.query))
        if k: refs.append((block.attention.self, 'key', block.attention.self.key))
        if v: refs.append((block.attention.self, 'value', block.attention.self.value))
        if attn_out: refs.append((block.attention.output, 'dense', block.attention.output.dense))
        if fc1: refs.append((block.intermediate, 'dense', block.intermediate.dense))
        if fc2: refs.append((block.output, 'dense', block.output.dense))
    return refs

# --------------- Param counting -----------------
@torch.no_grad()
def count_params(modules: List[torch.nn.Module]):
    return sum(m.weight.numel() for m in modules)

# --------------- Whitening-space truncation -----------------
def truncate_wspace(W: torch.Tensor, L: torch.Tensor, keep_ratio: float):
    Ww = W @ L
    U, S, Vt = torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
    rmax = S.numel(); r = max(1, min(rmax, math.ceil(keep_ratio * rmax)))
    U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]
    Ww_trunc = (U_r * S_r.unsqueeze(0)) @ Vt_r
    Wp = torch.linalg.solve_triangular(L.T, Ww_trunc.T, upper=True).T
    return Wp, r

@torch.no_grad()
def truncate_wspace_modules(modules: List[torch.nn.Module], L_list: List[torch.Tensor], keep_ratio: float, device: torch.device):
    assert len(modules) == len(L_list), f"Length mismatch: modules={len(modules)} vs L_list={len(L_list)}"
    kept = 0
    for mod_i, mod in enumerate(modules):
        L = L_list[mod_i].to(device); W = mod.weight.data.to(device)
        W_new, r = truncate_wspace(W, L, keep_ratio=keep_ratio)
        mod.weight.data.copy_(W_new); kept += r * (W.shape[0] + W.shape[1])
    return modules, kept

# --------------- Folding deltas (BERT compat) -----------------
@torch.no_grad()
def fold_in_deltas_bert(model, device, to_copy: bool = False):
    if bert_utils_legacy is None:
        raise RuntimeError("Legacy BERT utils unavailable; cannot fold deltas yet.")
    return bert_utils_legacy.fold_in_deltas(model, device, to_copy=to_copy)
