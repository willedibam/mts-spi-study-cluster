"""Plot the pilot/confirmation learning curves and independent-cohort contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


def plot(pilot, confirmation, output):
    reports = [json.loads(path.read_text()) for path in [pilot, confirmation]]
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Computer Modern Roman','CMU Serif','DejaVu Serif'], 'font.size':9,
        'axes.titlesize':10, 'legend.fontsize':8, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.grid':False, 'mathtext.fontset':'cm'})
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    methods = [('shape-pls','SPI distribution shapes','#b47b38'),
        ('z-pls',r'$z$ / PLS','#31688e'),
        ('random_encoder','Random encoder / PCA-ridge','#729a51'),
        ('linear','Calibrated linear dynamics','#222222')]
    bounds=np.asarray([report['primary'][method]['conditional_95_CI']
                       for report in reports for method,_,_ in methods])
    limits=(max(0,bounds.min()-.01),bounds.max()+.01)
    for ax,report,title in zip(axes[:2],reports,['Pilot: discrete settings','Confirmation: continuous settings']):
        labels=np.asarray(report['total_label_budgets'])
        for method,name,color in methods:
            values=report['primary'][method]
            ax.plot(labels,values['MAE'],'o-',color=color,lw=1.7,ms=3,label=name)
            if method in ['shape-pls','z-pls']:
                lo,hi=np.asarray(values['conditional_95_CI']).T
                ax.fill_between(labels,lo,hi,color=color,alpha=.13,lw=0)
        ax.set(xscale='log',xticks=labels,xticklabels=labels,
               xlabel='Labelled systems (including tuning)',ylabel='Joint-shift MAE',
               title=title,ylim=limits)
        ax.tick_params(direction='out')
        ax.xaxis.set_minor_locator(NullLocator())
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,frameon=False,loc='outside lower center',ncol=4)
    report=reports[1]; labels=np.asarray(report['total_label_budgets'])
    ax=axes[2]; ax.axhline(0,color='.65',lw=.8,ls='--')
    for family,marker,color,offset in [('linear','o','#31688e',-.11),('tanh','^','#729a51',.11)]:
        values=np.asarray(report['per_source_subset_primary_MAE']['z-pls'][family])-np.asarray(report['per_source_subset_primary_MAE']['shape-pls'][family])
        for i,row in enumerate(values):
            ax.scatter(i+offset+np.linspace(-.06,.06,len(row)),row,s=17,
                       color=color,marker=marker,label=family+' source' if i==0 else None)
    ax.set(xticks=np.arange(len(labels)),xticklabels=labels,
           xlabel='Labelled systems',ylabel=r'MAE difference: $z$ minus shapes',
           title='Confirmation: independent training cohorts')
    ax.legend(frameon=False,loc='best');ax.tick_params(direction='out')
    output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:
        fig.savefig(output.with_suffix('.'+extension),dpi=600,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['pilot','confirmation','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();plot(args.pilot,args.confirmation,args.output)
