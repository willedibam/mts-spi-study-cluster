import os
import _pickle as cPickle
import matplotlib.pyplot as plt
from utils import plotter
from umap import UMAP
import seaborn as sns
from degas import degas as dg

reducer = UMAP(random_state=42,min_dist=0.75,n_neighbors=9)

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise']
for d in datatypes:
    with open(os.path.join(resdir,d)+'.pkl','rb') as f:
        cf1 = cPickle.load(f)
        try:
            cf.merge(cf1)
        except NameError:
            cf = cf1

with open('database/pop-database.pkl','rb') as f:
    database = cPickle.load(f)

classes = [['coupled map lattice'],['var'],['oscillator'],['noise','cauchy'],['noise','normal']]


plt.rcParams.update({'font.size': 20})

cfplt = plotter(cf)

cmap = None

cfplt.dataspace(classes=classes,reducer=reducer,plot_nas=False,size='processes',
                    xlabel='UMAP-1',ylabel='UMAP-2',include_contour=True,cmap=cmap)
cfplt.fig.savefig(f'plots/5class_lowdim.pdf',dpi=300,bbox_inches='tight',transparent=True)

plt.show()

cfplt = plotter(cf,database=database)

fig, _ = cfplt.dataspace(classes=classes,reducer=reducer,plot_nas=False,size='processes',
                                xlabel='UMAP-1',ylabel='UMAP-2',include_contour=True)
fig.savefig(f'plots/5class_lowdim.pdf',dpi=300,bbox_inches='tight',transparent=True,cmap=cmap)

fig, _ = cfplt.dataspace(classes=classes,reducer=reducer,plot_nas=False,size='observations',
                                xlabel='UMAP-1',ylabel='UMAP-2',include_contour=True,cmap=cmap)

plt.show()