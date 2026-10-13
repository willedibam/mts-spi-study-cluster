from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/neurotycho_target_pilot_260910/evaluation')
r=json.loads((root/'report.json').read_text())
methods=['m-pca','mg-pca','z-pca','z-pls','mz-pca','validity-pca','spectrum','matched','enriched','pool']
labels=['SPI marginals + PCA','Marginals + graph + PCA','Pearson z + PCA','Pearson z + PLS','Marginals + z + PCA','Validity mask + PCA','Spectrum + logistic','Matched raw encoder','Enriched raw encoder','Learned SPI pooling']
fig,axes=plt.subplots(1,2,figsize=(11,5.5),sharey=True,layout='constrained')
for ax,m,title in zip(axes,[16,8],['16 bipolar channels · 8 s','8 bipolar channels · 4 s']):
 for i,method in enumerate(methods):
  ss=[s for s in r['scores'] if s['method']==method and s['M']==m]
  av=[np.mean([x['balanced_accuracy'] for s in ss for x in s['animals'] if x['animal']==a]) for a in ['Chibi','George']]
  ax.plot(av,[i,i],color='#c5c9d0',lw=1.5,zorder=1)
  for value,c in zip(av,['#2c70a9','#d0792a']):ax.scatter(value,i,c=c,s=30,zorder=3)
  ax.scatter(np.mean(av),i,marker='D',facecolors='white',edgecolors='#20252d',s=32,zorder=4)
 ax.set(title=title,xlim=(.47,1.035),xlabel='Balanced accuracy at fixed threshold 0.5')
 ax.set_xticks([.5,.6,.7,.8,.9,1.]);ax.grid(axis='x',alpha=.17)
 ax.spines[['top','right','left']].set_visible(False)
axes[0].set_yticks(range(len(methods)),labels);axes[0].invert_yaxis()
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for c,l in [('#2c70a9','Chibi (two dates)'),('#d0792a','George (two dates)')]]+[Line2D([],[],marker='D',ls='',markerfacecolor='white',color='#20252d',label='Mean of animals')],loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Frozen KTMD → propofol transfer pilot\nNeural points average three seeds; two target animals, no population uncertainty',fontsize=12)
fig.savefig(root/'transfer-by-animal.png',dpi=180);fig.savefig(root/'transfer-by-animal.svg')
