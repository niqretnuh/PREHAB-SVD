from __future__ import annotations
import math
import torch
import numpy as np
import evaluate

# ---------------- GLUE evaluation -----------------
@torch.no_grad()
def eval_glue_model(test_dl, model, task: str):
    """Evaluate a HuggingFace classification/regression model on a GLUE dataloader.
    Handles STS-B (regression) specially; others use argmax classification.
    Returns full metrics dict from evaluate.load("glue", task).
    """
    metric = evaluate.load("glue", task)
    model.eval()
    all_preds, all_labels = [], []
    regression = (task == "stsb")
    for batch in test_dl:
        # Expect already on device in caller or move lazily
        device = next(model.parameters()).device
        input_ids = torch.stack(batch["input_ids"], dim=1).to(device)
        attention_mask = torch.stack(batch["attention_mask"], dim=1).to(device)
        if regression:
            labels = batch["label"].float().to(device)
        else:
            labels = batch["label"].long().to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits.detach().cpu().numpy()
        all_preds.append(logits)
        all_labels.append(batch["label"].cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if regression:
        predictions = all_preds.squeeze()
    else:
        predictions = all_preds.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=all_labels)

# --------------- Image classification top-1 -----------------
@torch.no_grad()
def compute_top1(loader, model, device, amp: bool = False):
    """Compute top-1 accuracy for image classification model over a dataloader."""
    model.eval(); top1 = 0; total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp):
            logits = model(xb)
        top1 += (logits.argmax(dim=1) == yb).sum().item(); total += yb.size(0)
    return top1 / max(total, 1)

# --------------- Stable / effective rank surrogate ---------------
class _StableRankFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M: torch.Tensor, iters: int = 50):
        # Frobenius norm squared
        fro_sq = torch.trace(M.T @ M)
        # power iteration for spectral norm
        b = torch.randn(M.shape[1], dtype=M.dtype, device=M.device)
        b /= torch.linalg.norm(b)
        for _ in range(iters):
            b = M.T @ (M @ b)
            b /= torch.linalg.norm(b)
        spectral = torch.sqrt(b @ (M.T @ (M @ b)))
        spectral = torch.clamp(spectral, min=1e-10)
        ctx.save_for_backward(M, b)
        ctx.spectral = spectral
        ctx.fro_sq = fro_sq
        return fro_sq / (spectral ** 2)
    @staticmethod
    def backward(ctx, grad_output):
        M, b = ctx.saved_tensors
        spectral = ctx.spectral; fro_sq = ctx.fro_sq
        grad_fro = 2 * M
        u = (M @ b) / spectral
        grad_spectral = u.unsqueeze(1) @ b.unsqueeze(0)
        grad_M = grad_fro * spectral.pow(-2) + fro_sq * (-2 * spectral.pow(-3)) * grad_spectral
        return grad_output * grad_M, None

def stable_rank(M: torch.Tensor, iters: int = 50):
    return _StableRankFn.apply(M, iters)

@torch.no_grad()
def overall_effective_rank(modules, iters: int = 50):
    """Sum stable ranks over list of modules (expects .weight)."""
    acc = 0.0
    for mod in modules:
        W = mod.weight.data
        acc += stable_rank(W, iters=iters).item()
    return acc

# --------------- Helper for entropy-based effective rank ---------------
@torch.no_grad()
def entropy_effective_rank(W: torch.Tensor, eps: float = 1e-6):
    """Compute entropy-based effective rank exp(H(p)) where p are normalized singular values."""
    s = torch.linalg.svdvals(W.to(torch.float64))
    p = (s + eps) / (s.sum() + eps)
    h = -(p * (p + eps).log()).sum().item()
    return math.exp(h)
