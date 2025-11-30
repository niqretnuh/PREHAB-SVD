#!/usr/bin/env python3
"""
Unified whitening profiler script for BERT / ViT / Llama (placeholder).
Outputs whitening matrices to an output directory.
BERT: produces list of Cholesky factors per targeted Linear.
ViT: profiles qkv/proj/fc1/fc2/head.
Llama: placeholder.
"""
import os, json, argparse
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import timm
from torch.utils.data import DataLoader

from models.bert import utils as bert_utils
from models.vit.imagenet import load_imagenet

# (reuse helpers from unified prehab script if needed; duplicate minimal set here)
@torch.no_grad()
def chol_from_cov(C: torch.Tensor, eig_floor=1e-5, jitter_min=1e-6):
    C = 0.5*(C + C.T)
    try:
        return torch.linalg.cholesky(C.double())
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(C.double())
        evals = torch.clamp(evals, min=float(eig_floor))
        C2 = (evecs * evals.unsqueeze(0)) @ evecs.T
        try: return torch.linalg.cholesky(C2)
        except RuntimeError:
            I = torch.eye(C2.shape[0], device=C2.device, dtype=C2.dtype)
            return torch.linalg.cholesky(C2 + jitter_min * I)

@torch.no_grad()
def profile_whiteners_vit(model, loader, device, max_batches=None):
    m = model.to(device).eval(); num_blocks = len(m.blocks); D = m.embed_dim
    cov_qkv=[torch.zeros((D,D),device=device) for _ in range(num_blocks)]; n_qkv=[0]*num_blocks
    cov_fc1=[torch.zeros((D,D),device=device) for _ in range(num_blocks)]; n_fc1=[0]*num_blocks
    cov_fc2=[None for _ in range(num_blocks)]; n_fc2=[0]*num_blocks
    cov_proj=[None for _ in range(num_blocks)]; n_proj=[0]*num_blocks
    cov_head=None; n_head=0; hooks=[]
    for i,blk in enumerate(m.blocks):
        def h_norm1(idx):
            def h(_m,_in,out): X=out.detach().reshape(-1,out.shape[-1]); cov_qkv[idx].add_(X.T@X); n_qkv[idx]+=X.shape[0]
            return h
        hooks.append(blk.norm1.register_forward_hook(h_norm1(i)))
    for i,blk in enumerate(m.blocks):
        def h_norm2(idx):
            def h(_m,_in,out): X=out.detach().reshape(-1,out.shape[-1]); cov_fc1[idx].add_(X.T@X); n_fc1[idx]+=X.shape[0]
            return h
        hooks.append(blk.norm2.register_forward_hook(h_norm2(i)))
    for i,blk in enumerate(m.blocks):
        def h_act(idx):
            def h(_m,_in,out): X=out.detach().reshape(-1,out.shape[-1]); H=X.shape[-1];
            if cov_fc2[idx] is None: cov_fc2[idx]=torch.zeros((H,H),device=device); cov_fc2[idx].add_(X.T@X); n_fc2[idx]+=X.shape[0]
            return h
        hooks.append(blk.mlp.act.register_forward_hook(h_act(i)))
    for i,blk in enumerate(m.blocks):
        def pre_proj(idx):
            def pre(_m,inp): (X,)=inp; X=X.detach().reshape(-1,X.shape[-1]); C=X.shape[-1];
            if cov_proj[idx] is None: cov_proj[idx]=torch.zeros((C,C),device=device); cov_proj[idx].add_(X.T@X); n_proj[idx]+=X.shape[0]; return None
            return pre
        hooks.append(blk.attn.proj.register_forward_pre_hook(pre_proj(i)))
    head_is_linear=isinstance(getattr(m,'head',None),nn.Linear)
    if head_is_linear:
        def pre_head(_m,inp): (X,)=inp; X=X.detach().reshape(-1,X.shape[-1]); C=X.shape[-1];
            nonlocal cov_head,n_head
            if cov_head is None: cov_head=torch.zeros((C,C),device=device)
            cov_head.add_(X.T@X); n_head+=X.shape[0]; return None
        hooks.append(m.head.register_forward_pre_hook(pre_head))
    batches=0
    for xb,_ in loader:
        with torch.amp.autocast(device_type='cuda', enabled=False): m(xb.to(device,non_blocking=True))
        batches+=1
        if max_batches and batches>=max_batches: break
    for h in hooks: h.remove()
    L_qkv=[chol_from_cov(cov_qkv[i]/max(1,n_qkv[i])).float().cpu() for i in range(num_blocks)]
    L_fc1=[chol_from_cov(cov_fc1[i]/max(1,n_fc1[i])).float().cpu() for i in range(num_blocks)]
    L_fc2=[chol_from_cov(cov_fc2[i]/max(1,n_fc2[i])).float().cpu() for i in range(num_blocks)]
    L_proj=[chol_from_cov(cov_proj[i]/max(1,n_proj[i])).float().cpu() for i in range(num_blocks)]
    L_head=None
    if head_is_linear and cov_head is not None and n_head>0:
        L_head=chol_from_cov(cov_head/max(1,n_head)).float().cpu()
    return L_qkv,L_proj,L_fc1,L_fc2,L_head

# ===================== BERT WHITEN =====================

def run_bert_whiten(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cfg_path=os.path.join(args.output_dir,'whiten_config.json')
    out_path=os.path.join(args.output_dir,'whiten.pth')
    if args.skip_exists and os.path.exists(out_path):
        print(f"Skipping whitening | {out_path} exists"); return
    if not os.path.exists(args.input_model_path):
        raise FileNotFoundError(f"Missing finetuned model file: {args.input_model_path}")
    bert_utils.set_seed(args.seed+1)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(cfg_path,'w') as f: json.dump(vars(args), f, indent=2)
    train_loader=bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='train', num_workers=args.num_workers, shuffle=True, pin_memory=True)
    model=AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=bert_utils.GLUE_TO_NUMLABELS[args.glue_task])
    model.load_state_dict(torch.load(args.input_model_path), strict=True); model.to(device).eval()
    print(f"Profiling covariances for BERT using {args.whiten_nbatch * args.batch_size} samples …")
    modules=[module for _,_,module in bert_utils.get_module_references(model)]
    with torch.amp.autocast(device_type='cuda', enabled=args.amp):
        L_list=bert_utils.whiten_nbatch(model, train_loader, modules, device, nbatch=args.whiten_nbatch, jitter=args.jitter, zero=args.zero, regression=(args.glue_task=='stsb'))
        bert_utils.sanity_check_L(L_list)
    torch.save(L_list, out_path); print(f"Saved BERT whiteners → {out_path}")

# ===================== ViT WHITEN =====================

def run_vit_whiten(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cfg_path=os.path.join(args.output_dir,'whiten_config.json')
    with open(cfg_path,'w') as f: json.dump(vars(args), f, indent=2)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_ds,_=load_imagenet(args.imagenet_path, simple_augmentation=False)
    loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model=timm.create_model(args.vit_arch, pretrained=True).to(device).eval()
    print("Profiling ViT whiteners …")
    L_qkv,L_proj,L_fc1,L_fc2,L_head=profile_whiteners_vit(model, loader, device, max_batches=args.prof_batches)
    torch.save({'L_qkv':L_qkv,'L_proj':L_proj,'L_fc1':L_fc1,'L_fc2':L_fc2,'L_head':L_head}, os.path.join(args.output_dir,'whiten_vit.pth'))
    print(f"Saved ViT whiteners → {os.path.join(args.output_dir,'whiten_vit.pth')}")

# ===================== Llama Placeholder =====================

def run_llama_whiten(args):
    print('Llama whitening not implemented yet.')

# ===================== Parser / Dispatcher =====================

def build_parser():
    p=argparse.ArgumentParser(description='Unified whitening profiler')
    p.add_argument('--model', choices=['bert','vit','llama'], required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', default=False)
    # BERT specific
    p.add_argument('--glue_task', type=str, choices=['cola','mnli','mnli_matched','mnli_mismatched','mrpc','qnli','qqp','rte','sst2','stsb'])
    p.add_argument('--input_model_path', type=str)
    p.add_argument('--whiten_nbatch', type=int, default=100)
    p.add_argument('--jitter', type=float, default=1e-6)
    p.add_argument('--zero', type=float, default=1e-12)
    p.add_argument('--skip_exists', action='store_true', default=False)
    # ViT specific
    p.add_argument('--imagenet_path', type=str)
    p.add_argument('--vit_arch', type=str, default='vit_base_patch16_224')
    p.add_argument('--prof_batches', type=int, default=50)
    return p


def main():
    args=build_parser().parse_args()
    if args.model=='bert':
        needed=['glue_task','input_model_path']
        miss=[x for x in needed if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing BERT args: {miss}')
        run_bert_whiten(args)
    elif args.model=='vit':
        needed=['imagenet_path']
        miss=[x for x in needed if getattr(args,x) in (None,'')]
        if miss: raise ValueError(f'Missing ViT args: {miss}')
        run_vit_whiten(args)
    else:
        run_llama_whiten(args)

if __name__=='__main__':
    main()
