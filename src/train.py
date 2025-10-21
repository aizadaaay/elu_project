import argparse, os, json, csv, yaml, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .models import SmallCNN
from .data import get_loaders
from .utils import set_seed, accuracy
OPT={'adam':lambda p,lr,wd:optim.Adam(p,lr=lr,weight_decay=wd),'sgd':lambda p,lr,wd:optim.SGD(p,lr=lr,momentum=0.9,weight_decay=wd)}
def plot(xs,curves,title,ylabel,out):
 plt.figure();
 [plt.plot(xs,v,label=k) for k,v in curves.items()]; plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--activation',default='elu',choices=['elu','relu','leakyrelu'])
 ap.add_argument('--optimizer',default='adam',choices=list(OPT.keys()))
 ap.add_argument('--dropout',type=float,default=None); ap.add_argument('--subset_size',type=int,default=None)
 ap.add_argument('--bn',default='on',choices=['on','off']); ap.add_argument('--config',default='config.yaml'); a=ap.parse_args()
 cfg=yaml.safe_load(open(a.config)); set_seed(cfg.get('seed',42)); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 sub=a.subset_size if a.subset_size is not None else cfg.get('default_subset_size',10000)
 tr,va=get_loaders(cfg.get('data_dir','./data'),cfg.get('batch_size',128),cfg.get('num_workers',2),cfg.get('val_split',5000),sub)
 p=a.dropout if a.dropout is not None else cfg.get('default_dropout',0.2)
 m=SmallCNN(activation=a.activation,use_bn=(a.bn=='on'),p_dropout=p).to(dev)
 crit=nn.CrossEntropyLoss(); opt=OPT[a.optimizer](m.parameters(),lr=cfg.get('lr',1e-3),wd=cfg.get('weight_decay',1e-4))
 tag=f"{a.activation}_{a.optimizer}_{a.bn}_drop{p}_sub{sub}"; out=os.path.join('results',tag); os.makedirs(out,exist_ok=True)
 json.dump({'activation':a.activation,'optimizer':a.optimizer,'bn':a.bn,'dropout':p,'subset_size':sub,**cfg},open(os.path.join(out,'run_config.json'),'w'))
 hist=[]
 def one_epoch():
  m.train(); tl=ta=seen=0
  for x,y in tr:
   x,y=x.to(dev),y.to(dev); opt.zero_grad(); o=m(x); loss=crit(o,y); loss.backward(); opt.step(); b=x.size(0); tl+=loss.item()*b; ta+=accuracy(o.detach(),y)*b; seen+=b
  return tl/seen, ta/seen
 @torch.no_grad()
 def eval():
  m.eval(); vl=va=seen=0; Y=[]; P=[]
  for x,y in va:
   x,y=x.to(dev),y.to(dev); o=m(x); l=crit(o,y); b=x.size(0); vl+=l.item()*b; va+=accuracy(o,y)*b; seen+=b; Y.append(y.cpu().numpy()); P.append(o.argmax(1).cpu().numpy())
  Y=np.concatenate(Y); P=np.concatenate(P); val_loss=vl/seen; val_acc=va/seen; val_err=1.0-val_acc; return val_loss,val_acc,val_err,Y,P
 best=0.0
 for ep in range(1,cfg.get('epochs',10)+1):
  trl,tra=one_epoch(); vll,vaa,verr,Y,P=eval(); hist.append([ep,trl,vll,tra,vaa,verr]); print(f'Epoch {ep:02d} | val_acc={vaa:.4f} err={verr:.4f}')
  if vaa>best: best=vaa; torch.save(m.state_dict(), os.path.join(out,'best.pt'))
 csv.writer(open(os.path.join(out,'metrics.csv'),'w',newline='')).writerows([['epoch','train_loss','val_loss','train_acc','val_acc','val_err'],*hist])
 xs=[h[0] for h in hist]
 plot(xs,{'Train Loss':[h[1] for h in hist],'Val Loss':[h[2] for h in hist]},tag+' Loss','Loss',os.path.join(out,'loss_curve.png'))
 plot(xs,{'Train Acc':[h[3] for h in hist],'Val Acc':[h[4] for h in hist]},tag+' Acc','Accuracy',os.path.join(out,'acc_curve.png'))
 plot(xs,{'Val Error':[h[5] for h in hist]},tag+' Error','Error',os.path.join(out,'error_curve.png'))
 cm=confusion_matrix(Y,P,labels=list(range(10)))
 import matplotlib.pyplot as plt
 fig=plt.figure(); ax=fig.add_subplot(111); im=ax.imshow(cm,interpolation='nearest'); fig.colorbar(im); ax.set_title(tag+' Confusion'); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); plt.tight_layout(); plt.savefig(os.path.join(out,'confusion.png'),dpi=160); plt.close(fig)
 json.dump({'best_val_acc':best}, open(os.path.join(out,'summary.json'),'w'))
if __name__=='__main__': main()
