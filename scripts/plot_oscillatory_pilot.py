"""Compact publication-style view of the verified, originally specified pilot."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from src.representation_state_data import file_hash


def main():
    root=Path('results/oscillatory_coorganization_pilot_260909');report=root/'verified-report'
    frame=pd.read_csv(report/'per-fit.csv');summary=pd.read_csv(report/'summary.csv')
    contrasts=json.loads((report/'contrasts.json').read_text())['comparisons']
    styles=[('z-pls','z + PLS','#7030A0','-'),('m-pls','SPI marginals','#2077B4','-'),
            ('shape-pls','Normalized shapes','#559E9D','-'),('m+z-pls','Marginals + z','#C64D82','--'),
            ('raw:agreement-pls','Direct phase/envelope agreement','#BC7615','-'),
            ('neural-aligned','Aligned neural encoder','#777777',':')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.5),gridspec_kw={'width_ratios':[1,1,.9]})
    handles=[]
    for ax,m,title in zip(axes[:2],[16,8],['Original observation\nM = 16, T = 1000','Reduced observation\nM = 8, T = 500']):
        for method,label,color,style in styles:
            s=summary[(summary.method==method)&(summary.M==m)].sort_values('labels')
            assert s.labels.tolist()==[10,20,40] and (s.fits==3).all()
            line,=ax.plot(np.arange(3),s.balanced_accuracy,color=color,ls=style,marker='o',ms=4,lw=1.7,label=label)
            if m==16:handles.append(line)
            for offset,seed in zip([-.045,0,.045],[11,23,47]):
                values=frame[(frame.method==method)&(frame.M==m)&(frame.seed==seed)].sort_values('labels').balanced_accuracy
                ax.scatter(np.arange(3)+offset,values,s=10,color=color,alpha=.35,zorder=2)
        ax.axhline(.5,color='#aaaaaa',lw=.8,ls='--');ax.set_ylim(.4,1.025)
        ax.set_xticks([0,1,2],[10,20,40]);ax.set_xlabel('Labelled recordings');ax.set_title(title,fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Balanced accuracy')
    ax=axes[2]
    chosen=[next(c for c in contrasts if c['M']==8 and c['labels']==n and c['metric']=='balanced_accuracy'
                 and c['contrast']=='z-pls minus m-pls') for n in [10,20,40]]
    mean=np.array([c['mean'] for c in chosen])*100
    error=np.array([[c['mean']-c['low'] for c in chosen],[c['high']-c['mean'] for c in chosen]])*100
    ax.errorbar([0,1,2],mean,yerr=error,fmt='o',color='#7030A0',capsize=4,lw=1.5)
    wide=frame[frame.M==8].pivot(index=['labels','seed'],columns='method',values='balanced_accuracy')
    for offset,seed in zip([-.055,0,.055],[11,23,47]):
        values=wide.xs(seed,level='seed');ax.scatter(np.arange(3)+offset,100*(values['z-pls']-values['m-pls']),
                                                   color='#777777',s=15,alpha=.55)
    ax.axhline(0,color='#aaaaaa',ls='--',lw=.8);ax.set_ylim(0,40)
    ax.set_xticks([0,1,2],[10,20,40]);ax.set_xlabel('Labelled recordings')
    ax.set_title('Gain over SPI marginals\nReduced observation',fontsize=11)
    ax.set_ylabel('Percentage points');ax.grid(axis='y',alpha=.15)
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.045,.995),ncol=3,frameon=False,fontsize=9)
    fig.text(.765,.93,'Independent test recordings: 200\nThree disjoint training cohorts',fontsize=9,va='top')
    fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.75,wspace=.36)
    fig.text(.065,.065,'Lines: cohort means; small points: individual cohorts. Fixed decision threshold: 0.5.',fontsize=9)
    fig.text(.065,.025,'Intervals: paired test-master bootstrap, conditional on fitted models; pointwise 95%. All methods/scopes are in the report.',fontsize=8.5)
    for suffix in ['png','svg']:fig.savefig(root/f'learning-curves.{suffix}',dpi=180,bbox_inches='tight')
    (root/'learning-curves.json').write_text(json.dumps(dict(script_sha256=file_hash(Path(__file__)),
        source_sha256={p:file_hash(report/p) for p in ['per-fit.csv','summary.csv','contrasts.json']}),indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':main()
