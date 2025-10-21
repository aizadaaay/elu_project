import random, numpy as np, torch
def set_seed(s=42):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s);
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def accuracy(logits,y):
 return (logits.argmax(1)==y).float().mean().item()
