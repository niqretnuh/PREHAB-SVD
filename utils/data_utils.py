"""Shared data loading utilities.
Migrates logic from models/bert/utils.py and models/vit/imagenet.py.

Functions:
  get_glue_dataloader(task_name, batch_size=32, split='train', **kwargs)
  load_imagenet(path, simple_augmentation=False)  (delegates to models.vit.imagenet)
  build_tokenizer(model_name='bert-base-uncased')
"""
from __future__ import annotations
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
try:
    from models.vit.imagenet import load_imagenet as _vit_load_imagenet
except Exception:
    _vit_load_imagenet = None

GLUE_TO_KEYS = {
    'cola': ('sentence', None),
    'mnli_matched': ('premise', 'hypothesis'),
    'mnli_mismatched': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

def build_tokenizer(model_name: str = 'bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def get_glue_dataloader(task_name: str, batch_size: int = 32, split: str = 'train', **kwargs):
    """Return PyTorch DataLoader for a GLUE task.
    Automatically handles MNLI matched/mismatched split naming.
    """
    if task_name.startswith('mnli'):
        raw = load_dataset('glue', 'mnli')
        split_name = f"{split}_{task_name.split('_')[-1]}" if split != 'train' else split
    else:
        raw = load_dataset('glue', task_name)
        split_name = split
    tokenizer = build_tokenizer()
    key1, key2 = GLUE_TO_KEYS[task_name]
    def tokenize(batch):
        if key2 is None:
            return tokenizer(batch[key1], padding='max_length', truncation=True, max_length=128)
        return tokenizer(batch[key1], batch[key2], padding='max_length', truncation=True, max_length=128)
    tok = raw.map(tokenize, batched=True)
    return torch.utils.data.DataLoader(tok[split_name], batch_size=batch_size, **kwargs)

def load_imagenet(path: str, simple_augmentation: bool = False):
    if _vit_load_imagenet is None:
        raise RuntimeError('ViT ImageNet loader not available in this environment.')
    return _vit_load_imagenet(path, simple_augmentation=simple_augmentation)

# ...end of file...
