#!/usr/bin/env python3
"""
Unified SVD threshold sweep script (compression without whitening) for BERT / ViT / Llama placeholder.
BERT: apply plain SVD rank truncation to targeted linear modules.
ViT: apply SVD to qkv/proj/fc1/fc2/head weights (optional split qkv).
Llama: placeholder.
Outputs CSV of threshold -> accuracy / kept params percentage.
"""
import os, math, copy, json, argparse
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForSequenceClassification
import timm
from torch.utils.data import DataLoader

from models.bert import utils as bert_utils
from models.vit.imagenet import load_imagenet

# ===================== Helpers =====================

def _print(msg):
    print(msg)

@torch.no_grad()
def _svd_truncate(W: torch.Tensor, keep_ratio: float):
    U,S,Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    rmax = S.numel(); r = max(1, min(rmax, math.ceil(keep_ratio * rmax)))
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    Wt = (U_r * S_r.unsqueeze(0)) @ Vh_r
    kept = r * (W.shape[0] + W.shape[1])
    return Wt.to(W.dtype), kept

# ===================== BERT SVD SWEEP =====================
@torch.no_grad()
def run_bert_svd(args):
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=bert_utils.GLUE_TO_NUMLABELS[args.glue_task])
    model.load_state_dict(torch.load(args.input_model_path), strict=True)
    model.to(device).eval()
    test_loader = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='validation', num_workers=args.num_workers, pin_memory=True)
    orig_params = bert_utils.count_params(model)
    results = {1.0:(None,100.0)}
    modules_refs = bert_utils.get_module_references(model)
    for th in args.thresholds:
        m = copy.deepcopy(model)
        kept_total = 0
        for parent, attr, mod in bert_utils.get_module_references(m):
            W = mod.weight.data.to(device)
            Wt, kept = _svd_truncate(W, th)
            mod.weight.data.copy_(Wt)
            kept_total += kept
        acc = bert_utils.eval_model(test_loader, m, args.glue_task)[bert_utils.GLUE_TO_MAIN_METRIC[args.glue_task]]
        kept_pct = 100.0 * kept_total / orig_params
        _print(f"[BERT SVD] th={th:.2f} acc={acc:.4f} kept={kept_pct:.2f}%")
        results[th]=(acc, kept_pct)
    _save_csv(args.output_csv, results)

# ===================== ViT SVD SWEEP =====================
@torch.no_grad()
def _eval_vit(loader, model, device, amp):
    model.eval(); cor=tot=0
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp):
            logits=model(x)
        cor += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
    return cor/max(1,tot)

@torch.no_grad()
def run_vit_svd(args):
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds = load_imagenet(args.imagenet_path, simple_augmentation=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base = timm.create_model(args.vit_arch, pretrained=True).to(device).eval()
    # param count over targeted modules
    def count_mod_params(m):
        tot=0
        for blk in m.blocks:
            tot += blk.attn.qkv.weight.numel()
            tot += blk.attn.proj.weight.numel()
            tot += blk.mlp.fc1.weight.numel()
            tot += blk.mlp.fc2.weight.numel()
        if isinstance(getattr(m,'head',None), nn.Linear):
            tot += m.head.weight.numel()
        return tot
    orig_params=count_mod_params(base)
    results={1.0:(None,100.0)}
    for th in args.thresholds:
        m=copy.deepcopy(base).to(device).eval(); kept_total=0
        for blk in m.blocks:
            # qkv
            Wqkv = blk.attn.qkv.weight.data.to(device)
            if args.split_qkv:
                assert Wqkv.shape[0] % 3 == 0
                D=Wqkv.shape[0]//3
                Ws=[Wqkv[:D], Wqkv[D:2*D], Wqkv[2*D:]]
                new_parts=[]
                for W in Ws:
                    Wt, kept=_svd_truncate(W, th); new_parts.append(Wt); kept_total+=kept
                blk.attn.qkv.weight.data.copy_(torch.cat(new_parts, dim=0))
            else:
                Wt, kept = _svd_truncate(Wqkv, th); blk.attn.qkv.weight.data.copy_(Wt); kept_total+=kept
            # proj
            Wp=blk.attn.proj.weight.data.to(device); Wt, kept=_svd_truncate(Wp, th); blk.attn.proj.weight.data.copy_(Wt); kept_total+=kept
            # fc1
            W1=blk.mlp.fc1.weight.data.to(device); Wt, kept=_svd_truncate(W1, th); blk.mlp.fc1.weight.data.copy_(Wt); kept_total+=kept
            # fc2
            W2=blk.mlp.fc2.weight.data.to(device); Wt, kept=_svd_truncate(W2, th); blk.mlp.fc2.weight.data.copy_(Wt); kept_total+=kept
        if isinstance(getattr(m,'head',None), nn.Linear):
            Wh=m.head.weight.data.to(device); Wt, kept=_svd_truncate(Wh, th); m.head.weight.data.copy_(Wt); kept_total+=kept
        top1=_eval_vit(val_loader, m, device, args.amp)
        kept_pct=100.0*kept_total/orig_params
        _print(f"[ViT SVD] th={th:.2f} top1={top1*100:.2f}% kept={kept_pct:.2f}%")
        results[th]=(top1, kept_pct)
    _save_csv(args.output_csv, results)

# ===================== Llama placeholder =====================
@torch.no_grad()
def run_llama_svd(args):
    _print('Llama SVD not implemented yet.')
    results={1.0:(None,100.0)}
    _save_csv(args.output_csv, results)

# ===================== CSV =====================

def _save_csv(path, res_dict):
    data=[]
    for th,(acc,kept) in res_dict.items():
        data.append({'threshold':th,'accuracy':acc,'params_kept_pct':kept})
    df=pd.DataFrame(data)
    df['threshold']=df['threshold'].astype(float)
    df=df.sort_values(['threshold'], ascending=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format='%.6f')
    _print(f"Saved results → {path}")

# ===================== Parser / Dispatcher =====================

def build_parser():
    p=argparse.ArgumentParser(description='Unified SVD sweep')
    p.add_argument('--model', choices=['bert','vit','llama'], required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
    p.add_argument('--output_csv', type=str, required=True)
    p.add_argument('--amp', action='store_true', default=False)  # ViT eval
    # BERT specific
    p.add_argument('--glue_task', type=str, choices=['cola','mnli','mnli_matched','mnli_mismatched','mrpc','qnli','qqp','rte','sst2','stsb'])
    p.add_argument('--input_model_path', type=str)
    # ViT specific
    p.add_argument('--imagenet_path', type=str)
    p.add_argument('--vit_arch', type=str, default='vit_base_patch16_224')
    p.add_argument('--split_qkv', action='store_true', default=False)
    return p


def main():
    args=build_parser().parse_args()
    torch.manual_seed(args.seed)
    if args.model=='bert':
        miss=[x for x in ['glue_task','input_model_path'] if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing BERT args: {miss}')
        run_bert_svd(args)
    elif args.model=='vit':
        miss=[x for x in ['imagenet_path'] if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing ViT args: {miss}')
        run_vit_svd(args)
    else:
        run_llama_svd(args)

if __name__=='__main__':
    main()
