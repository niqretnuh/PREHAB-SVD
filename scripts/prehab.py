#!/usr/bin/env python3
"""
Unified PREHAB script supporting:
  - BERT (GLUE tasks)  (whitened low-rank regularization in whitened space)
  - ViT  (ImageNet)    (full-model whitening + nuclear norm surrogate)
  - Llama (placeholder)

Usage examples:
  BERT:
    python scripts/prehab.py --model bert \
        --glue_task sst2 --input_model_path path/to/finetuned.pth \
        --input_whiten_path path/to/whiten.pth --output_dir out/bert_prehab

  ViT:
    python scripts/prehab.py --model vit \
        --imagenet_path /path/to/imagenet --output_dir out/vit_prehab \
        --epochs 2500 --prof_batches 50 --split_qkv

  Llama:
    python scripts/prehab.py --model llama   (currently not implemented)
"""
import os, math, json, sys, copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# BERT dependencies
from transformers import AutoModelForSequenceClassification
# Keep using existing BERT utils for now (migration later)
from models.bert import utils as bert_utils

# ViT dependencies
import timm
from torch.utils.data import DataLoader
from models.vit.imagenet import load_imagenet  # ImageNet loader

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (can later be moved to utils/):
# ──────────────────────────────────────────────────────────────────────────────

def _print_flush(msg, file=None):
    print(msg)
    if file:
        file.write(str(msg) + "\n"); file.flush()

# ====== VI T HELPERS (ported from models/vit/prehab.py) ======================
@torch.no_grad()
def robust_svd(A: torch.Tensor, full_matrices: bool = False):
    dev = A.device
    A32 = A.float()
    try:
        return torch.linalg.svd(A32, full_matrices=full_matrices, driver='gesvdj')
    except Exception:
        pass
    A64 = A32.cpu().to(torch.float64)
    U, S, Vh = torch.linalg.svd(A64, full_matrices=full_matrices)
    return U.to(dev, torch.float32), S.to(dev, torch.float32), Vh.to(dev, torch.float32)

@torch.no_grad()
def chol_from_cov(C: torch.Tensor, eig_floor=1e-5, jitter_min=1e-6):
    C = 0.5*(C + C.T)
    try:
        return torch.linalg.cholesky(C.double())
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(C.double())
        evals = torch.clamp(evals, min=float(eig_floor))
        C2 = (evecs * evals.unsqueeze(0)) @ evecs.T
        try:
            return torch.linalg.cholesky(C2)
        except RuntimeError:
            I = torch.eye(C2.shape[0], device=C2.device, dtype=C2.dtype)
            return torch.linalg.cholesky(C2 + jitter_min * I)

# Numerics for ViT nuclear norm surrogate
@torch.no_grad()
def svdvals_safe(M: torch.Tensor, ridge: float = 1e-8, cpu_fallback: bool = True) -> torch.Tensor:
    dev = M.device
    with torch.amp.autocast(device_type='cuda', enabled=False):
        try:
            if M.shape[0] >= M.shape[1]:
                G = M.T @ M
            else:
                G = M @ M.T
            G = 0.5 * (G + G.T)
            tr = torch.trace(G)
            eps = ridge * (tr / G.shape[0] if torch.isfinite(tr) else 1.0)
            G = G + eps * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
            evals, _ = torch.linalg.eigh(G, UPLO='U')
            evals = torch.clamp(evals, min=0.0)
            s = torch.sqrt(evals)
            s = torch.flip(s, dims=[0])
            return s.to(torch.float32)
        except Exception:
            pass
        if cpu_fallback:
            if M.shape[0] >= M.shape[1]:
                G = M.T @ M
            else:
                G = M @ M.T
            G = 0.5 * (G + G.T)
            G = G.to('cpu', torch.float64)
            tr = torch.trace(G)
            eps = ridge * (tr / G.shape[0] if torch.isfinite(tr) else 1.0)
            G = G + eps * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
            evals, _ = torch.linalg.eigh(G, UPLO='U')
            evals = torch.clamp(evals, min=0.0)
            s = torch.sqrt(evals)
            s = torch.flip(s, dims=[0]).to(dev, torch.float32)
            return s
        raise RuntimeError("svdvals_safe failed on both GPU and CPU.")

def nuclear_norm_surrogate(M: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    m, n = M.shape
    G = (M.T @ M) if m >= n else (M @ M.T)
    G = 0.5 * (G + G.T)
    tr = torch.trace(G)
    eps = ridge * (tr / G.shape[0] if torch.isfinite(tr) else 1.0)
    G = G + eps * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    evals, _ = torch.linalg.eigh(G)
    s = torch.sqrt(evals.clamp_min(0))
    return s.sum()

# Wrapper modules for ViT
class FullDeltaLinear(nn.Module):
    def __init__(self, Ww_orig, L, bias):
        super().__init__()
        self.register_buffer('Ww_orig', Ww_orig)
        self.register_buffer('L', L)
        if bias is None:
            bias = torch.zeros(Ww_orig.shape[0], device=Ww_orig.device, dtype=Ww_orig.dtype)
        self.register_buffer('bias_orig', bias)
        self.delta_Ww = nn.Parameter(torch.zeros_like(Ww_orig))
    def forward(self, x):
        Ww_p = self.Ww_orig + self.delta_Ww
        Wt = torch.linalg.solve_triangular(self.L.T, Ww_p.T, upper=True)
        W = Wt.T
        return F.linear(x, W, bias=self.bias_orig)

class FullDeltaQKVSplit(nn.Module):
    def __init__(self, Ww_q, Ww_k, Ww_v, L, bias_qkv):
        super().__init__()
        self.register_buffer('Ww_q', Ww_q)
        self.register_buffer('Ww_k', Ww_k)
        self.register_buffer('Ww_v', Ww_v)
        self.register_buffer('L', L)
        if bias_qkv is None:
            D = Ww_q.shape[0]
            bias_qkv = torch.zeros(3*D, device=Ww_q.device, dtype=Ww_q.dtype)
        self.register_buffer('bias_qkv', bias_qkv)
        self.delta_q = nn.Parameter(torch.zeros_like(Ww_q))
        self.delta_k = nn.Parameter(torch.zeros_like(Ww_k))
        self.delta_v = nn.Parameter(torch.zeros_like(Ww_v))
    def _solve(self, Ww):
        Wt = torch.linalg.solve_triangular(self.L.T, Ww.T, upper=True)
        return Wt.T
    def forward(self, x):
        Wq = self._solve(self.Ww_q + self.delta_q)
        Wk = self._solve(self.Ww_k + self.delta_k)
        Wv = self._solve(self.Ww_v + self.delta_v)
        W = torch.cat([Wq, Wk, Wv], dim=0)
        return F.linear(x, W, bias=self.bias_qkv)

@torch.no_grad()
def fold_in_deltas_vit(student, device):
    for i, blk in enumerate(student.blocks):
        # qkv
        if isinstance(blk.attn.qkv, FullDeltaQKVSplit):
            m = blk.attn.qkv
            Wq = torch.linalg.solve_triangular(m.L.T, (m.Ww_q + m.delta_q).T, upper=True).T
            Wk = torch.linalg.solve_triangular(m.L.T, (m.Ww_k + m.delta_k).T, upper=True).T
            Wv = torch.linalg.solve_triangular(m.L.T, (m.Ww_v + m.delta_v).T, upper=True).T
            W = torch.cat([Wq, Wk, Wv], dim=0)
            new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
            new.weight.data.copy_(W); new.bias.data.copy_(m.bias_qkv)
            blk.attn.qkv = new
        elif isinstance(blk.attn.qkv, FullDeltaLinear):
            m = blk.attn.qkv
            W = torch.linalg.solve_triangular(m.L.T, (m.Ww_orig + m.delta_Ww).T, upper=True).T
            new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
            new.weight.data.copy_(W); new.bias.data.copy_(m.bias_orig)
            blk.attn.qkv = new
        # proj
        if isinstance(blk.attn.proj, FullDeltaLinear):
            m = blk.attn.proj
            W = torch.linalg.solve_triangular(m.L.T, (m.Ww_orig + m.delta_Ww).T, upper=True).T
            new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
            new.weight.data.copy_(W); new.bias.data.copy_(m.bias_orig)
            blk.attn.proj = new
        # fc1
        if isinstance(blk.mlp.fc1, FullDeltaLinear):
            m = blk.mlp.fc1
            W = torch.linalg.solve_triangular(m.L.T, (m.Ww_orig + m.delta_Ww).T, upper=True).T
            new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
            new.weight.data.copy_(W); new.bias.data.copy_(m.bias_orig)
            blk.mlp.fc1 = new
        # fc2
        if isinstance(blk.mlp.fc2, FullDeltaLinear):
            m = blk.mlp.fc2
            W = torch.linalg.solve_triangular(m.L.T, (m.Ww_orig + m.delta_Ww).T, upper=True).T
            new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
            new.weight.data.copy_(W); new.bias.data.copy_(m.bias_orig)
            blk.mlp.fc2 = new
    # head
    if isinstance(getattr(student, "head", None), FullDeltaLinear):
        m = student.head
        W = torch.linalg.solve_triangular(m.L.T, (m.Ww_orig + m.delta_Ww).T, upper=True).T
        new = nn.Linear(W.shape[1], W.shape[0], bias=True).to(device)
        new.weight.data.copy_(W); new.bias.data.copy_(m.bias_orig)
        student.head = new

@torch.no_grad()
def profile_whiteners_vit(model, loader, device, max_batches=None):
    m = model.to(device).eval()
    num_blocks = len(m.blocks)
    D = m.embed_dim
    cov_qkv  = [torch.zeros((D, D), device=device) for _ in range(num_blocks)]; n_qkv  = [0]*num_blocks
    cov_fc1  = [torch.zeros((D, D), device=device) for _ in range(num_blocks)]; n_fc1  = [0]*num_blocks
    cov_fc2  = [None for _ in range(num_blocks)]; n_fc2  = [0]*num_blocks
    cov_proj = [None for _ in range(num_blocks)]; n_proj = [0]*num_blocks
    cov_head = None; n_head = 0
    hooks = []
    for i, blk in enumerate(m.blocks):
        def make_norm1(idx):
            def h(_m, _inp, out):
                X = out.detach().reshape(-1, out.shape[-1]); cov_qkv[idx].add_(X.T @ X); n_qkv[idx] += X.shape[0]
            return h
        hooks.append(blk.norm1.register_forward_hook(make_norm1(i)))
    for i, blk in enumerate(m.blocks):
        def make_norm2(idx):
            def h(_m, _inp, out):
                X = out.detach().reshape(-1, out.shape[-1]); cov_fc1[idx].add_(X.T @ X); n_fc1[idx] += X.shape[0]
            return h
        hooks.append(blk.norm2.register_forward_hook(make_norm2(i)))
    for i, blk in enumerate(m.blocks):
        def make_act(idx):
            def h(_m, _inp, out):
                X = out.detach().reshape(-1, out.shape[-1]); H = X.shape[-1]
                if cov_fc2[idx] is None: cov_fc2[idx] = torch.zeros((H, H), device=device)
                cov_fc2[idx].add_(X.T @ X); n_fc2[idx] += X.shape[0]
            return h
        hooks.append(blk.mlp.act.register_forward_hook(make_act(i)))
    for i, blk in enumerate(m.blocks):
        def make_pre_proj(idx):
            def pre(_m, inp):
                (X,) = inp; X = X.detach().reshape(-1, X.shape[-1]); C = X.shape[-1]
                if cov_proj[idx] is None: cov_proj[idx] = torch.zeros((C, C), device=device)
                cov_proj[idx].add_(X.T @ X); n_proj[idx] += X.shape[0]
                return None
            return pre
        hooks.append(blk.attn.proj.register_forward_pre_hook(make_pre_proj(i)))
    head_is_linear = isinstance(getattr(m, 'head', None), nn.Linear)
    if head_is_linear:
        def pre_head(_m, inp):
            (X,) = inp; X = X.detach().reshape(-1, X.shape[-1]); C = X.shape[-1]
            nonlocal cov_head, n_head
            if cov_head is None: cov_head = torch.zeros((C, C), device=device)
            cov_head.add_(X.T @ X); n_head += X.shape[0]
            return None
        hooks.append(m.head.register_forward_pre_hook(pre_head))
    batches = 0
    for xb, _ in loader:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            m(xb.to(device, non_blocking=True))
        batches += 1
        if max_batches and batches >= max_batches: break
    for h in hooks: h.remove()
    L_qkv  = [chol_from_cov(cov_qkv[i] / max(1, n_qkv[i])).float().cpu() for i in range(num_blocks)]
    L_fc1  = [chol_from_cov(cov_fc1[i] / max(1, n_fc1[i])).float().cpu() for i in range(num_blocks)]
    L_fc2  = [chol_from_cov(cov_fc2[i] / max(1, n_fc2[i])).float().cpu() for i in range(num_blocks)]
    L_proj = [chol_from_cov(cov_proj[i] / max(1, n_proj[i])).float().cpu() for i in range(num_blocks)]
    L_head = None
    if head_is_linear and cov_head is not None and n_head > 0:
        L_head = chol_from_cov(cov_head / max(1, n_head)).float().cpu()
    return L_qkv, L_proj, L_fc1, L_fc2, L_head

# ====== BERT WRAPPER (from models/bert/prehab.py) ============================
class BertFullDeltaFC(nn.Module):
    def __init__(self, Ww_orig, L, bias):
        super().__init__()
        self.register_buffer('Ww_orig', Ww_orig)
        self.register_buffer('L', L)
        self.register_buffer('bias_orig', bias)
        self.delta_Ww = nn.Parameter(torch.zeros_like(Ww_orig))
    def forward(self, x):
        Ww_p = self.Ww_orig + self.delta_Ww
        Wt = torch.linalg.solve_triangular(self.L.T, Ww_p.T, upper=True)
        W = Wt.T
        return F.linear(x, W, bias=self.bias_orig)

@torch.no_grad()
def sanity_check_wrappers_bert(model):
    bad = []
    wrapped_modules = bert_utils.get_wrapped_modules(model)
    module_names = ['query', 'key', 'value', 'attn_out', 'fc1', 'fc2']
    for i, module in enumerate(wrapped_modules):
        layer_idx = i // len(module_names)
        module_idx = i % len(module_names)
        Wt = torch.linalg.solve_triangular(module.L.T, module.Ww_orig.T, upper=True)
        if not torch.isfinite(Wt).all(): bad.append((layer_idx, module_names[module_idx]))
    return bad

# ====== PREHAB IMPLEMENTATIONS ==============================================

def run_bert_prehab(args):
    # Environment & paths
    os.makedirs(args.output_dir, exist_ok=True)
    prehab_model_path = os.path.join(args.output_dir, 'prehab.pth')
    prehab_config     = os.path.join(args.output_dir, 'prehab_config.json')
    prehab_metric     = os.path.join(args.output_dir, 'prehab_metric.json')
    prehab_log        = os.path.join(args.output_dir, 'prehab_log.txt')
    if args.skip_exists and os.path.exists(prehab_model_path):
        _print_flush(f"Skipping prehab | {prehab_model_path} exists")
        return
    if not os.path.exists(args.input_model_path):
        raise FileNotFoundError(f"Missing finetuned model file: {args.input_model_path}")
    if not os.path.exists(args.input_whiten_path):
        raise FileNotFoundError(f"Missing whitening file: {args.input_whiten_path}")
    bert_utils.set_seed(args.seed + 2)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(prehab_config, 'w') as f: json.dump(vars(args), f, indent=2)
    log_f = open(prehab_log, 'w'); log_f.write('step,phase,loss_obj,loss_nuc,erank\n')
    steps, ces, nucs, eranks = [], [], [], []
    def record(step, loss_obj, loss_nuc, erank):
        steps.append(step)
        ces.append(loss_obj.item() if loss_obj is not None else float('nan'))
        nucs.append(loss_nuc.item() if loss_nuc is not None else float('nan'))
        eranks.append(erank if isinstance(erank, float) else float(erank.item()))
        phase = 'obj' if loss_obj is not None else 'rank'
        log_f.write(f"{step},{phase},{ces[-1]},{nucs[-1]},{eranks[-1]}\n")
    # Data
    train_loader = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='train', num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_loader  = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='validation', num_workers=args.num_workers, pin_memory=True)
    # Model
    teacher = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=bert_utils.GLUE_TO_NUMLABELS[args.glue_task]).to(device).eval()
    teacher.load_state_dict(torch.load(args.input_model_path), strict=True)
    student = AutoModelForSequenceClassification.from_config(teacher.config).to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    # Load whiteners
    L_list = torch.load(args.input_whiten_path, map_location='cpu')
    bert_utils.sanity_check_L(L_list)
    modules = [module for _, _, module in bert_utils.get_module_references(student)]
    WW_list, L_use, orig_params = [], [], 0
    for (mod, L_cpu) in zip(modules, L_list):
        L = L_cpu.to(device); W = mod.weight.data.to(device)
        WW_list.append(W @ L); L_use.append(L); orig_params += W.numel()
    # Replace with wrappers
    module_refs = bert_utils.get_module_references(student)
    for i, (parent, attr_name, module) in enumerate(module_refs):
        bias_orig = module.bias.data.to(device) if module.bias is not None else torch.zeros(WW_list[i].shape[0], device=device)
        setattr(parent, attr_name, BertFullDeltaFC(Ww_orig=WW_list[i], L=L_use[i], bias=bias_orig))
    bad = sanity_check_wrappers_bert(student)
    if bad: raise RuntimeError(f"Non-finite unwhitening in blocks: {bad}")
    # Freeze
    for p in student.parameters(): p.requires_grad = False
    wrapped_modules = bert_utils.get_wrapped_modules(student)
    delta_params = []
    for module in wrapped_modules:
        module.delta_Ww.requires_grad = True; delta_params.append(module.delta_Ww)
    # Initial ranks
    num_modules = len(modules)
    initial_w_erank = bert_utils.overallAll_erank(teacher, iters=args.stableRank_iters) / num_modules
    initial_ww_erank = 0.0
    for module in wrapped_modules:
        Ww = module.Ww_orig.to(torch.float64)
        erank = bert_utils.stable_rank(Ww, iters=args.stableRank_iters)
        initial_ww_erank += erank.item()
    initial_ww_erank /= num_modules
    opt = optim.Adam(delta_params, lr=args.lr)
    train_iter = iter(train_loader)
    student.train()
    if args.prehab_nbatch is None: args.prehab_nbatch = len(train_loader)
    lambda_schedule = [args.lambda_initial * math.exp((math.log(args.lambda_max/args.lambda_initial)/args.lambda_warmup)*epoch) if epoch < args.lambda_warmup else args.lambda_max for epoch in range(args.prehab_nbatch)]
    _print_flush(f"Number of prehab batches (BERT): {args.prehab_nbatch}")
    for epoch in range(1, args.prehab_nbatch + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)
        output = bert_utils.forward_pass(student, batch, device=device, regression=(args.glue_task == 'stsb'))
        loss_obj = output.loss
        loss_rank = torch.zeros((), device=device, dtype=torch.float64)
        wrapped_modules = bert_utils.get_wrapped_modules(student)
        for module in wrapped_modules:
            Ww_p = (module.Ww_orig + module.delta_Ww).to(torch.float64)
            erank = bert_utils.stable_rank(Ww_p, iters=args.stableRank_iters)
            loss_rank = loss_rank + erank
        loss_rank = loss_rank / num_modules
        current_ww_erank = loss_rank.item()
        loss = loss_obj + (lambda_schedule[epoch-1] * loss_rank.to(torch.float32))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        record(epoch, loss_obj, (lambda_schedule[epoch-1] * loss_rank.to(torch.float32)), current_ww_erank)
        if epoch == 1 or epoch % 10 == 0:
            _print_flush(f"[{epoch}/{args.prehab_nbatch}] Rank={(lambda_schedule[epoch-1] * loss_rank).item():.3e} Obj={loss_obj.item():.3e} Er={current_ww_erank:.3f} Lambda={lambda_schedule[epoch-1]:.3f} | WL Erank ratio={current_ww_erank/initial_ww_erank:.3f}")
            if args.checkpoint:
                folded_model = bert_utils.fold_in_deltas(student, device, to_copy=True)
                current_w_erank = bert_utils.overallAll_erank(folded_model, iters=args.stableRank_iters) / num_modules
                w_erank_ratio = current_w_erank / initial_w_erank
                ww_erank_ratio = current_ww_erank / initial_ww_erank
                _print_flush(f"\t[Epoch {epoch}] WL effective rank ratio={ww_erank_ratio:.3f} | W effective rank ratio={w_erank_ratio:.3f}", file=log_f)
    with torch.no_grad():
        bert_utils.fold_in_deltas(student, device)
    final_w_erank = bert_utils.overallAll_erank(student, iters=args.stableRank_iters) / num_modules
    _print_flush(f"[Final] W effective rank = {final_w_erank:.3f} | ratio = {final_w_erank/initial_w_erank:.3f}")
    torch.save(student.state_dict(), prehab_model_path)
    log_f.close(); _print_flush(f"Training log saved to {prehab_log}")
    acc = bert_utils.eval_model(test_loader, student, args.glue_task)
    _print_flush(f"Final performance: {acc}")
    res = {
        'final_performance': acc[bert_utils.GLUE_TO_MAIN_METRIC[args.glue_task]],
        'initial_w_erank': initial_w_erank,
        'final_w_erank': final_w_erank,
        'final_w_erank_ratio': final_w_erank / initial_w_erank,
    }
    with open(prehab_metric, 'w') as f: json.dump(res, f, indent=2)
    _print_flush(f"Metrics saved to {prehab_metric}")


def run_vit_prehab(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    log_f = open(os.path.join(args.output_dir, 'training_log.txt'), 'w')
    log_f.write('step,phase,loss_ce,loss_nuc,erank\n')
    steps, ces, nucs, eranks = [], [], [], []
    def record(step, loss_ce, loss_nuc, erank):
        steps.append(step)
        ces.append(loss_ce.item() if loss_ce is not None else float('nan'))
        nucs.append(loss_nuc.item() if loss_nuc is not None else float('nan'))
        eranks.append(erank if isinstance(erank, float) else float(erank))
        phase = 'ce' if loss_ce is not None else 'rank'
        log_f.write(f"{step},{phase},{ces[-1]},{nucs[-1]},{eranks[-1]}\n"); log_f.flush()
    # Data
    train_ds, test_ds = load_imagenet(args.imagenet_path, simple_augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    prof_loader  = DataLoader(train_ds, batch_size=args.batch_whiten, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # Teacher
    teacher = timm.create_model(args.vit_arch, pretrained=True).to(device).eval()
    _print_flush('Profiling whiteners for ViT…')
    L_qkv, L_proj, L_fc1, L_fc2, L_head = profile_whiteners_vit(teacher, prof_loader, device, max_batches=args.prof_batches)
    student = timm.create_model(args.vit_arch, pretrained=False).to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    orig_params = 0; n_rank_terms = 0; delta_params = []
    for i, blk in enumerate(student.blocks):
        # qkv
        Wqkv = blk.attn.qkv.weight.data.to(device)
        bqkv = blk.attn.qkv.bias.data.to(device) if blk.attn.qkv.bias is not None else None
        Lq = L_qkv[i].to(device)
        if args.split_qkv:
            assert Wqkv.shape[0] % 3 == 0
            D = Wqkv.shape[0] // 3
            Wq, Wk, Wv = Wqkv[:D], Wqkv[D:2*D], Wqkv[2*D:]
            blk.attn.qkv = FullDeltaQKVSplit(Wq @ Lq, Wk @ Lq, Wv @ Lq, Lq, bqkv)
            delta_params += [blk.attn.qkv.delta_q, blk.attn.qkv.delta_k, blk.attn.qkv.delta_v]
            n_rank_terms += 3; orig_params += (Wq.numel() + Wk.numel() + Wv.numel())
        else:
            blk.attn.qkv = FullDeltaLinear(Wqkv @ Lq, Lq, bqkv)
            delta_params += [blk.attn.qkv.delta_Ww]; n_rank_terms += 1; orig_params += Wqkv.numel()
        # proj
        Wp = blk.attn.proj.weight.data.to(device)
        bp = blk.attn.proj.bias.data.to(device) if blk.attn.proj.bias is not None else None
        Lp = L_proj[i].to(device)
        blk.attn.proj = FullDeltaLinear(Wp @ Lp, Lp, bp)
        delta_params += [blk.attn.proj.delta_Ww]; n_rank_terms += 1; orig_params += Wp.numel()
        # fc1
        W1 = blk.mlp.fc1.weight.data.to(device)
        b1 = blk.mlp.fc1.bias.data.to(device) if blk.mlp.fc1.bias is not None else None
        L1 = L_fc1[i].to(device)
        blk.mlp.fc1 = FullDeltaLinear(W1 @ L1, L1, b1)
        delta_params += [blk.mlp.fc1.delta_Ww]; n_rank_terms += 1; orig_params += W1.numel()
        # fc2
        W2 = blk.mlp.fc2.weight.data.to(device)
        b2 = blk.mlp.fc2.bias.data.to(device) if blk.mlp.fc2.bias is not None else None
        L2 = L_fc2[i].to(device)
        blk.mlp.fc2 = FullDeltaLinear(W2 @ L2, L2, b2)
        delta_params += [blk.mlp.fc2.delta_Ww]; n_rank_terms += 1; orig_params += W2.numel()
    # head
    if isinstance(getattr(student, 'head', None), nn.Linear) and (L_head is not None):
        Wh = student.head.weight.data.to(device)
        bh = student.head.bias.data.to(device) if student.head.bias is not None else None
        Lh = L_head.to(device)
        student.head = FullDeltaLinear(Wh @ Lh, Lh, bh)
        delta_params += [student.head.delta_Ww]; n_rank_terms += 1; orig_params += Wh.numel()
    # Freeze
    for p in student.parameters(): p.requires_grad = False
    for p in delta_params: p.requires_grad = True
    opt = optim.Adam(delta_params, lr=args.lr)
    ce_crit = nn.CrossEntropyLoss()
    # Train loop
    train_iter = iter(train_loader)
    for epoch in range(args.epochs):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            logits = student(x); loss_ce = ce_crit(logits, y)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            loss_nuc = torch.zeros((), device=device, dtype=torch.float64); er_acc = 0.0
            for blk in student.blocks:
                if isinstance(blk.attn.qkv, FullDeltaQKVSplit):
                    for Ww in (blk.attn.qkv.Ww_q + blk.attn.qkv.delta_q, blk.attn.qkv.Ww_k + blk.attn.qkv.delta_k, blk.attn.qkv.Ww_v + blk.attn.qkv.delta_v):
                        Ww = Ww.to(torch.float64)
                        loss_nuc = loss_nuc + nuclear_norm_surrogate(Ww, ridge=1e-6)
                        s = svdvals_safe(Ww)
                        p = (s + 1e-6) / (s.sum() + 1e-6)
                        h = -(p * (p + 1e-6).log()).sum().item()
                        er_acc += math.exp(h) * (Ww.shape[0] + Ww.shape[1])
                else:
                    Ww = (blk.attn.qkv.Ww_orig + blk.attn.qkv.delta_Ww).to(torch.float64)
                    loss_nuc = loss_nuc + nuclear_norm_surrogate(Ww, ridge=1e-6)
                    s = svdvals_safe(Ww)
                    p = (s + 1e-6) / (s.sum() + 1e-6)
                    h = -(p * (p + 1e-6).log()).sum().item()
                    er_acc += math.exp(h) * (Ww.shape[0] + Ww.shape[1])
                Wwp = (blk.attn.proj.Ww_orig + blk.attn.proj.delta_Ww).to(torch.float64)
                loss_nuc = loss_nuc + nuclear_norm_surrogate(Wwp, ridge=1e-6)
                sp = svdvals_safe(Wwp); pp = (sp + 1e-6) / (sp.sum() + 1e-6); hp = -(pp * (pp + 1e-6).log()).sum().item(); er_acc += math.exp(hp) * (Wwp.shape[0] + Wwp.shape[1])
                Ww1 = (blk.mlp.fc1.Ww_orig + blk.mlp.fc1.delta_Ww).to(torch.float64)
                loss_nuc = loss_nuc + nuclear_norm_surrogate(Ww1, ridge=1e-6)
                s1 = svdvals_safe(Ww1); p1 = (s1 + 1e-6) / (s1.sum() + 1e-6); h1 = -(p1 * (p1 + 1e-6).log()).sum().item(); er_acc += math.exp(h1) * (Ww1.shape[0] + Ww1.shape[1])
                Ww2 = (blk.mlp.fc2.Ww_orig + blk.mlp.fc2.delta_Ww).to(torch.float64)
                loss_nuc = loss_nuc + nuclear_norm_surrogate(Ww2, ridge=1e-6)
                s2 = svdvals_safe(Ww2); p2 = (s2 + 1e-6) / (s2.sum() + 1e-6); h2 = -(p2 * (p2 + 1e-6).log()).sum().item(); er_acc += math.exp(h2) * (Ww2.shape[0] + Ww2.shape[1])
            if isinstance(getattr(student, 'head', None), FullDeltaLinear):
                Wwh = (student.head.Ww_orig + student.head.delta_Ww).to(torch.float64)
                loss_nuc = loss_nuc + nuclear_norm_surrogate(Wwh, ridge=1e-6)
                sh = svdvals_safe(Wwh); ph = (sh + 1e-6) / (sh.sum() + 1e-6); hh = -(ph * (ph + 1e-6).log()).sum().item(); er_acc += math.exp(hh) * (Wwh.shape[0] + Wwh.shape[1])
            loss_nuc = loss_nuc / float(n_rank_terms); erank = er_acc / float(orig_params)
        loss = loss_ce + (args.lambda_rank * loss_nuc.to(torch.float32))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        record(epoch, loss_ce, (args.lambda_rank * loss_nuc.to(torch.float32)), erank)
        if epoch % 10 == 0 or epoch == 1:
            _print_flush(f"[ViT {epoch}/{args.epochs}] Nuc={(args.lambda_rank*loss_nuc).item():.3e} CE={loss_ce.item():.3e} Er={erank:.3f}")
    with torch.no_grad():
        fold_in_deltas_vit(student, device)
    # Eval
    student.eval(); top1 = 0; tot = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            logits = student(x)
        top1 += (logits.argmax(dim=1) == y).sum().item(); tot += y.size(0)
    acc = top1 / max(1, tot)
    _print_flush(f"Final ViT top-1: {acc*100:.2f}%")
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'final_model_folded.pth'))
    meta = {
        'arch': args.vit_arch,
        'lr': args.lr,
        'lambda_rank': args.lambda_rank,
        'epochs': args.epochs,
        'orig_params': int(orig_params),
        'amp': args.amp,
        'split_qkv': args.split_qkv,
        'prof_batches': args.prof_batches,
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f: json.dump(meta, f, indent=2)
    log_f.close(); _print_flush(f"Training log: {os.path.join(args.output_dir, 'training_log.txt')}")


def run_llama_prehab(_args):
    _print_flush('Llama PREHAB not implemented yet.')

# ====== ARGUMENT PARSING =====================================================

def build_parser():
    p = argparse.ArgumentParser(description='Unified PREHAB script (BERT / ViT / Llama placeholder)')
    p.add_argument('--model', type=str, choices=['bert', 'vit', 'llama'], required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    # BERT-specific
    p.add_argument('--glue_task', type=str, choices=['cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb'])
    p.add_argument('--input_model_path', type=str, help='Finetuned BERT model path')
    p.add_argument('--input_whiten_path', type=str, help='Whitening matrices path (BERT)')
    p.add_argument('--prehab_nbatch', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lambda_max', type=float, default=10.0)
    p.add_argument('--lambda_initial', type=float, default=1e-4)
    p.add_argument('--lambda_warmup', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--stableRank_iters', type=int, default=50)
    p.add_argument('--checkpoint', action='store_true', default=False)
    p.add_argument('--skip_exists', action='store_true', default=False)
    # Shared output dir
    p.add_argument('--output_dir', type=str, required=True)
    # ViT-specific
    p.add_argument('--imagenet_path', type=str, help='Path to ImageNet (ViT)')
    p.add_argument('--vit_arch', type=str, default='vit_base_patch16_224')
    p.add_argument('--epochs', type=int, default=2500)
    p.add_argument('--lambda_rank', type=float, default=5.0)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--split_qkv', action='store_true', default=False)
    p.add_argument('--prof_batches', type=int, default=50)
    p.add_argument('--batch_whiten', type=int, default=64)
    return p


def main():
    parser = build_parser(); args = parser.parse_args()
    if args.model == 'bert':
        required = ['glue_task', 'input_model_path', 'input_whiten_path']
        missing = [r for r in required if getattr(args, r) in (None, '')]
        if missing:
            raise ValueError(f"Missing required BERT args: {missing}")
        run_bert_prehab(args)
    elif args.model == 'vit':
        required = ['imagenet_path']
        missing = [r for r in required if getattr(args, r) in (None, '')]
        if missing:
            raise ValueError(f"Missing required ViT args: {missing}")
        run_vit_prehab(args)
    elif args.model == 'llama':
        run_llama_prehab(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()
