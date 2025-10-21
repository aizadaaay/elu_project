import argparse,glob,os,pandas as pd,matplotlib.pyplot as plt
ap=argparse.ArgumentParser(); ap.add_argument('--glob',default='results/*/metrics.csv'); ap.add_argument('--out',default='results/summary'); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
rows=[]
for p in glob.glob(a.glob):
 tag=os.path.basename(os.path.dirname(p)); df=pd.read_csv(p); rows.append({'run':tag,'best_val_acc':df['val_acc'].max()})
comp=pd.DataFrame(rows).sort_values('best_val_acc',ascending=False).reset_index(drop=True); comp.to_csv(os.path.join(a.out,'comparison.csv'),index=False)
plt.figure(); plt.bar(comp['run'],comp['best_val_acc']); plt.xticks(rotation=30,ha='right'); plt.ylabel('Best Val Acc'); plt.title('Best Validation Accuracy by Run'); plt.tight_layout(); plt.savefig(os.path.join(a.out,'comparison.png'),dpi=160)
