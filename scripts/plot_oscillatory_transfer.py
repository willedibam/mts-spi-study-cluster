"""Matched learning curves before and after the prespecified dynamics shift."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np,pandas as pd
from src.representation_state_data import file_hash


def main():
    source=Path('results/oscillatory_coorganization_pilot_260909/pca-pooling-report')
    root=Path('results/oscillatory_coorganization_transfer_260909');target=root/'verified-report'
    styles=[('z-pca','z + PCA/ridge','#7030A0','-'),('z-pls','z + PLS','#AA78BA','--'),
            ('learned-pooling','Learned SPI pooling','#247CA4','-'),('m-pls','Individual-SPI summaries','#C07835','-'),
            ('raw:agreement-pls','Direct phase/envelope agreement','#555555',':')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),sharex=True,sharey=True);handles=[]
    for row,(directory,name) in enumerate([(source,'Original dynamics'),(target,'Faster dynamics')]):
        s=pd.read_csv(directory/'summary.csv');f=pd.read_csv(directory/'per-fit.csv')
        for col,m in enumerate([16,8]):
            ax=axes[row,col]
            for method,label,color,style in styles:
                a=s[(s.M==m)&(s.method==method)].sort_values('labels')
                assert a.labels.tolist()==[10,20,40] and (a.fits==3).all()
                line,=ax.plot(np.arange(3),a.balanced_accuracy,color=color,ls=style,marker='o',lw=1.6,ms=4,label=label)
                if row==col==0:handles.append(line)
                for shift,seed in zip([-.035,0,.035],[11,23,47]):
                    values=f[(f.M==m)&(f.method==method)&(f.seed==seed)].sort_values('labels').balanced_accuracy
                    ax.scatter(np.arange(3)+shift,values,color=color,s=10,alpha=.35)
            ax.axhline(.5,color='#aaaaaa',ls='--',lw=.8);ax.grid(axis='y',alpha=.15)
            ax.set_title(f'{name}: M = {m}, T = {1000 if m==16 else 500}',fontsize=11)
            ax.set_xticks([0,1,2],[10,20,40]);ax.set_ylim(.43,1.025);ax.yaxis.set_major_formatter(PercentFormatter(1))
            if col==0:ax.set_ylabel('Balanced accuracy')
            if row==1:ax.set_xlabel('Labelled source recordings')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.97,top=.86,bottom=.14,hspace=.32,wspace=.16)
    fig.text(.09,.065,'Lines: means of three disjoint source cohorts; small points: individual cohorts. Fixed threshold: 0.5.',fontsize=9)
    fig.text(.09,.03,'All models use original-dynamics training data only. Each regime has 200 independent held-out test recordings.',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(root/f'learning-curves.{suffix}',dpi=180,bbox_inches='tight')
    (root/'learning-curves.json').write_text(json.dumps({'script_sha256':file_hash(Path(__file__)),
        'inputs':{str(p):file_hash(p) for d in [source,target] for p in [d/'summary.csv',d/'per-fit.csv']}},indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':main()
