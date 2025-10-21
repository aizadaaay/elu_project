import argparse,itertools,subprocess,sys
ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml');
ap.add_argument('--activations',nargs='+',default=['elu','relu','leakyrelu']); ap.add_argument('--optimizers',nargs='+',default=['adam','sgd']);
ap.add_argument('--bn',nargs='+',default=['on','off']); ap.add_argument('--dropouts',nargs='+',default=['0.0','0.2','0.5']); ap.add_argument('--subset_sizes',nargs='+',default=['5000','10000']);
a=ap.parse_args()
for act,opt,bn,drop,sub in itertools.product(a.activations,a.optimizers,a.bn,a.dropouts,a.subset_sizes):
 print(f'=== {act} {opt} {bn} drop{drop} sub{sub} ===')
 subprocess.run([sys.executable,'-m','src.train','--activation',act,'--optimizer',opt,'--bn',bn,'--dropout',str(drop),'--subset_size',str(sub),'--config',a.config],check=True)
