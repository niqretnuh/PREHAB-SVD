#!/usr/bin/env python3
"""
Unified finetune script for BERT / ViT / Llama (placeholder).
BERT: GLUE tasks fine-tuning.
ViT: ImageNet top-1 fine-tuning (linear or full). Minimal implementation.
Llama: placeholder.
"""
import os, json, math, argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
import timm
from torch.utils.data import DataLoader

# Reuse existing BERT utilities
from models.bert import utils as bert_utils
# ViT dataset loader
from models.vit.imagenet import load_imagenet


def _print(msg):
    print(msg)

# ===================== BERT FINETUNE =====================
def run_bert_finetune(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cfg_path = os.path.join(args.output_dir, 'finetune_config.json')
    metrics_path = os.path.join(args.output_dir, 'finetune_metrics.json')
    bert_utils.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(cfg_path, 'w') as f: json.dump(vars(args), f, indent=2)
    train_loader = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='train', num_workers=args.num_workers, shuffle=True)
    val_loader   = bert_utils.get_dataloader(args.glue_task, batch_size=args.batch_size, split='validation', num_workers=args.num_workers)
    num_labels = bert_utils.GLUE_TO_NUMLABELS[args.glue_task]
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
    if args.grad_ckpt: model.gradient_checkpointing_enable()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)
    best_metric = -1.0; best_state = None
    for ep in range(1, args.epochs+1):
        model.train(); running = 0.0; steps = 0
        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            out = bert_utils.forward_pass(model, batch, device=device, regression=(args.glue_task=='stsb'))
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step+1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); steps += 1
            running += loss.item()*args.grad_accum
        sched.step()
        _print(f"[BERT] Epoch {ep}/{args.epochs} train_loss={running/max(1,steps):.4f}")
        val_metrics = bert_utils.eval_model(val_loader, model, args.glue_task)
        main_metric = val_metrics[bert_utils.GLUE_TO_MAIN_METRIC[args.glue_task]]
        _print(f"[BERT] Val metrics: {val_metrics}")
        if main_metric > best_metric:
            best_metric = main_metric; best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.output_dir, 'model.pth'))
            with open(metrics_path, 'w') as f: json.dump(val_metrics, f, indent=2)
            _print(f"[BERT] New best {best_metric:.4f} saved.")
    _print(f"[BERT] Finished. Best={best_metric:.4f}")

# ===================== ViT FINETUNE =====================
@torch.no_grad()
def _vit_eval(loader, model, device, amp):
    model.eval(); cor=tot=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.amp.autocast(device_type='cuda', enabled=amp):
            logits=model(x)
        cor += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
    return cor/max(1,tot)

def run_vit_finetune(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cfg_path = os.path.join(args.output_dir,'finetune_config.json')
    with open(cfg_path,'w') as f: json.dump(vars(args), f, indent=2)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds = load_imagenet(args.imagenet_path, simple_augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = timm.create_model(args.vit_arch, pretrained=True).to(device)
    # Optionally just train classifier head
    if args.linear_probe:
        for n,p in model.named_parameters():
            if 'head' not in n: p.requires_grad=False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best = -1.0
    for ep in range(1, args.epochs+1):
        model.train(); epoch_loss=0.0; steps=0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                logits = model(x)
                loss = criterion(logits,y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); steps+=1; epoch_loss+=loss.item()
        sched.step()
        top1 = _vit_eval(val_loader, model, device, args.amp)
        _print(f"[ViT] Epoch {ep}/{args.epochs} loss={epoch_loss/max(1,steps):.4f} val_top1={top1*100:.2f}%")
        if top1>best:
            best=top1
            torch.save(model.state_dict(), os.path.join(args.output_dir,'model.pth'))
            with open(os.path.join(args.output_dir,'finetune_metrics.json'),'w') as f: json.dump({'top1':top1}, f, indent=2)
            _print(f"[ViT] New best {best*100:.2f}% saved.")
    _print(f"[ViT] Finished. Best top-1={best*100:.2f}%")

# ===================== Llama Placeholder =====================
def run_llama_finetune(args):
    _print('Llama finetune not implemented yet.')

# ===================== Parser / Dispatcher =====================

def build_parser():
    p = argparse.ArgumentParser(description='Unified finetune script')
    p.add_argument('--model', choices=['bert','vit','llama'], required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, required=True)
    # BERT
    p.add_argument('--glue_task', type=str, choices=['cola','mnli','mnli_matched','mnli_mismatched','mrpc','qnli','qqp','rte','sst2','stsb'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--grad_ckpt', action='store_true', default=False)
    # ViT
    p.add_argument('--imagenet_path', type=str, help='Path to ImageNet root (ViT)')
    p.add_argument('--vit_arch', type=str, default='vit_base_patch16_224')
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--linear_probe', action='store_true', default=False)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    return p


def main():
    args = build_parser().parse_args()
    if args.model == 'bert':
        missing = [x for x in ['glue_task'] if getattr(args,x) in (None,'')]
        if missing: raise ValueError(f'Missing BERT args: {missing}')
        run_bert_finetune(args)
    elif args.model == 'vit':
        missing = [x for x in ['imagenet_path'] if getattr(args,x) in (None,'')]
        if missing: raise ValueError(f'Missing ViT args: {missing}')
        run_vit_finetune(args)
    else:
        run_llama_finetune(args)

if __name__ == '__main__':
    main()
