#!/usr/bin/env python3
"""
Unified whitening-threshold sweep (apply truncation in whitened space) for BERT / ViT / Llama placeholder.
BERT: reuse whitening matrices (whiten.pth) and sweep keep ratios.
ViT: expects whiten_vit.pth with dict of L_*; performs rank truncation in whitened space analogous to BERT.
Llama: placeholder.
"""
import os, math, copy, argparse, json
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForSequenceClassification
import timm
from torch.utils.data import DataLoader

from models.bert import utils as bert_utils
from models.vit.imagenet import load_imagenet

# ===================== BERT whiten-space truncation =====================
@torch.no_grad()
def _truncate_wspace_modules(modules, L_list, keep_ratio, device):
    kept=0
    for mod, L in zip(modules, L_list):
        W = mod.weight.data.to(device)
        Ld = L.to(device)
        Ww = W @ Ld
        U,S,Vh = torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
        rmax=S.numel(); r=max(1,min(rmax, math.ceil(keep_ratio*rmax)))
        U_r,S_r,Vh_r = U[:, :r], S[:r], Vh[:r, :]
        Ww_trunc=(U_r * S_r.unsqueeze(0)) @ Vh_r
        # solve back: W = Ww_trunc * L^{-1}
        Wt = torch.linalg.solve_triangular(Ld.T, Ww_trunc.T, upper=True).T
        mod.weight.data.copy_(Wt.to(W.dtype))
        kept += r * (W.shape[0] + W.shape[1])
    return kept

@torch.no_grad()
def run_bert_whiten_thresholds(args):
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=bert_utils.GLUE_TO_NUMLABELS[args.glue_task])
    model.load_state_dict(torch.load(args.input_model_path), strict=True)
    model.to(device).eval()
    L_list = torch.load(args.whiten_path)
    bert_utils.sanity_check_L(L_list)
    test_loader = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='validation', num_workers=args.num_workers, pin_memory=True)
    orig_params = bert_utils.count_params(model)
    results={1.0:(None,100.0)}
    for th in args.thresholds:
        m=copy.deepcopy(model)
        modules=[module for _,_,module in bert_utils.get_module_references(m)]
        kept=_truncate_wspace_modules(modules, L_list, th, device)
        acc=bert_utils.eval_model(test_loader, m, args.glue_task)[bert_utils.GLUE_TO_MAIN_METRIC[args.glue_task]]
        kept_pct=100.0*kept/orig_params
        print(f"[BERT WSPACE] th={th:.2f} acc={acc:.4f} kept={kept_pct:.2f}%")
        results[th]=(acc, kept_pct)
    _save_csv(args.output_csv, results)

# ===================== ViT whiten-space truncation =====================
@torch.no_grad()
def _truncate_wspace_vit(model, L_dict, keep_ratio, device, split_qkv):
    kept=0
    for i, blk in enumerate(model.blocks):
        # qkv
        Wqkv = blk.attn.qkv.weight.data.to(device)
        Lq = L_dict['L_qkv'][i].to(device)
        if split_qkv:
            D=Wqkv.shape[0]//3
            Ws=[Wqkv[:D], Wqkv[D:2*D], Wqkv[2*D:]]
            new_parts=[]
            for W in Ws:
                Ww=W @ Lq; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
                r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
                Ww_trunc=(U[:,:r] * S[:r].unsqueeze(0)) @ Vh[:r,:]
                Wt=torch.linalg.solve_triangular(Lq.T, Ww_trunc.T, upper=True).T
                new_parts.append(Wt.to(W.dtype)); kept += r*(W.shape[0]+W.shape[1])
            blk.attn.qkv.weight.data.copy_(torch.cat(new_parts, dim=0))
        else:
            Ww=Wqkv @ Lq; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
            r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
            Ww_trunc=(U[:,:r] * S[:r].unsqueeze(0)) @ Vh[:r,:]
            Wt=torch.linalg.solve_triangular(Lq.T, Ww_trunc.T, upper=True).T
            blk.attn.qkv.weight.data.copy_(Wt.to(Wqkv.dtype)); kept += r*(Wqkv.shape[0]+Wqkv.shape[1])
        # proj
        Wp=blk.attn.proj.weight.data.to(device); Lp=L_dict['L_proj'][i].to(device)
        Ww=Wp @ Lp; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
        r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
        Ww_trunc=(U[:,:r]*S[:r].unsqueeze(0))@Vh[:r,:]
        Wt=torch.linalg.solve_triangular(Lp.T, Ww_trunc.T, upper=True).T
        blk.attn.proj.weight.data.copy_(Wt.to(Wp.dtype)); kept += r*(Wp.shape[0]+Wp.shape[1])
        # fc1
        W1=blk.mlp.fc1.weight.data.to(device); L1=L_dict['L_fc1'][i].to(device)
        Ww=W1 @ L1; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
        r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
        Ww_trunc=(U[:,:r]*S[:r].unsqueeze(0))@Vh[:r,:]
        Wt=torch.linalg.solve_triangular(L1.T, Ww_trunc.T, upper=True).T
        blk.mlp.fc1.weight.data.copy_(Wt.to(W1.dtype)); kept += r*(W1.shape[0]+W1.shape[1])
        # fc2
        W2=blk.mlp.fc2.weight.data.to(device); L2=L_dict['L_fc2'][i].to(device)
        Ww=W2 @ L2; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
        r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
        Ww_trunc=(U[:,:r]*S[:r].unsqueeze(0))@Vh[:r,:]
        Wt=torch.linalg.solve_triangular(L2.T, Ww_trunc.T, upper=True).T
        blk.mlp.fc2.weight.data.copy_(Wt.to(W2.dtype)); kept += r*(W2.shape[0]+W2.shape[1])
    if isinstance(getattr(model,'head',None), nn.Linear) and L_dict['L_head'] is not None:
        Wh=model.head.weight.data.to(device); Lh=L_dict['L_head'].to(device)
        Ww=Wh @ Lh; U,S,Vh=torch.linalg.svd(Ww.to(torch.float64), full_matrices=False)
        r=max(1,min(S.numel(), math.ceil(keep_ratio*S.numel())))
        Ww_trunc=(U[:,:r]*S[:r].unsqueeze(0))@Vh[:r,:]
        Wt=torch.linalg.solve_triangular(Lh.T, Ww_trunc.T, upper=True).T
        model.head.weight.data.copy_(Wt.to(Wh.dtype)); kept += r*(Wh.shape[0]+Wh.shape[1])
    return kept

@torch.no_grad()
def run_vit_whiten_thresholds(args):
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    val_ds = load_imagenet(args.imagenet_path, simple_augmentation=False)[1]
    val_loader=DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base=timm.create_model(args.vit_arch, pretrained=True).to(device).eval()
    L_dict=torch.load(args.whiten_path)
    def count_params(m):
        tot=0
        for blk in m.blocks:
            tot += blk.attn.qkv.weight.numel(); tot+=blk.attn.proj.weight.numel(); tot+=blk.mlp.fc1.weight.numel(); tot+=blk.mlp.fc2.weight.numel()
        if isinstance(getattr(m,'head',None), nn.Linear): tot+=m.head.weight.numel()
        return tot
    orig_params=count_params(base); results={1.0:(None,100.0)}
    for th in args.thresholds:
        m=copy.deepcopy(base).to(device).eval(); kept=_truncate_wspace_vit(m, L_dict, th, device, args.split_qkv)
        top1=_eval_vit(val_loader, m, device, args.amp)
        kept_pct=100.0*kept/orig_params
        print(f"[ViT WSPACE] th={th:.2f} top1={top1*100:.2f}% kept={kept_pct:.2f}%")
        results[th]=(top1, kept_pct)
    _save_csv(args.output_csv, results)

@torch.no_grad()
def _eval_vit(loader, model, device, amp):
    model.eval(); cor=tot=0
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp): logits=model(x)
        cor += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
    return cor/max(1,tot)

# ===================== CSV =====================

def _save_csv(path, res_dict):
    data=[]
    for th,(acc,kept) in res_dict.items():
        data.append({'threshold':th,'accuracy':acc,'params_kept_pct':kept})
    df=pd.DataFrame(data); df['threshold']=df['threshold'].astype(float); df=df.sort_values(['threshold'], ascending=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format='%.6f'); print(f"Saved results → {path}")

# ===================== Parser / Dispatcher =====================

def build_parser():
    p=argparse.ArgumentParser(description='Unified whiten-space threshold sweep')
    p.add_argument('--model', choices=['bert','vit','llama'], required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
    p.add_argument('--output_csv', type=str, required=True)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--split_qkv', action='store_true', default=False)
    # BERT
    p.add_argument('--glue_task', type=str, choices=['cola','mnli','mnli_matched','mnli_mismatched','mrpc','qnli','qqp','rte','sst2','stsb'])
    p.add_argument('--input_model_path', type=str)
    p.add_argument('--whiten_path', type=str)
    # ViT
    p.add_argument('--imagenet_path', type=str)
    p.add_argument('--vit_arch', type=str, default='vit_base_patch16_224')
    return p


def main():
    args=build_parser().parse_args(); torch.manual_seed(args.seed)
    if args.model=='bert':
        need=['glue_task','input_model_path','whiten_path']
        miss=[x for x in need if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing BERT args: {miss}')
        run_bert_whiten_thresholds(args)
    elif args.model=='vit':
        need=['imagenet_path','whiten_path']
        miss=[x for x in need if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing ViT args: {miss}')
        run_vit_whiten_thresholds(args)
    else:
        print('Llama whitening threshold sweep not implemented.')

if __name__=='__main__':
    main()
