import torch
import torch.nn as nn
def get_activation(n):
 n=n.lower();
 return nn.ELU(inplace=True) if n=='elu' else nn.ReLU(inplace=True) if n=='relu' else nn.LeakyReLU(0.01,inplace=True)
class SmallCNN(nn.Module):
 def __init__(self,activation='elu',use_bn=True,p_dropout=0.2):
  super().__init__(); A=lambda:get_activation(activation)
  self.features=nn.Sequential(nn.Conv2d(1,32,3,padding=1), (nn.BatchNorm2d(32) if use_bn else nn.Identity()), A(), nn.MaxPool2d(2),
                              nn.Conv2d(32,64,3,padding=1), (nn.BatchNorm2d(64) if use_bn else nn.Identity()), A(), nn.MaxPool2d(2))
  self.classifier=nn.Sequential(nn.Flatten(), nn.Linear(64*7*7,128), A(), nn.Dropout(p_dropout), nn.Linear(128,10))
 def forward(self,x): x=self.features(x); return self.classifier(x)
