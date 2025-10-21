import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
def get_loaders(data_dir,batch_size,num_workers,val_split=5000,subset_size=None):
 t=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,),(0.3530,))])
 full=datasets.FashionMNIST(root=data_dir,train=True,download=True,transform=t)
 trn_sz=len(full)-val_split
 tr,va=torch.utils.data.random_split(full,[trn_sz,val_split],generator=torch.Generator().manual_seed(123))
 if subset_size:
  tr=Subset(tr, list(range(min(subset_size,len(tr)))))
  va=Subset(va, list(range(max(min(subset_size//2,len(va)),1000))))
 return DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=num_workers), DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=num_workers)
