import os
import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import seaborn as sns

plt.rcParams.update({'font.size': 24})
flipped_axes = True

cls_to_pkl = {'coupled map lattice': 'cml_sample.pkl',
                'var': 'var_sample.pkl',
                'oscillator': 'kuramoto_sample.pkl',
                'cauchy': 'cauchy_sample.pkl',
                'normal': 'normal_sample.pkl'}

# classes = [['normal'],['var']]
# spis = ('cce_gaussian','cce_kozachenko')
# spis = ('sgc_parametric_mean_fs-1_fmin-0_fmax-0-5_order-1','sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1')

classes = [['normal'],['cauchy']]
spis = ('cov_EmpiricalCovariance','spearmanr')
# spis = ('dcorr','dcorr_biased')
# spis = ('pdist_cityblock', 'pdist_euclidean')

# classes = [['normal'],['coupled map lattice']]
# spis = ('tlmi_kraskov_NN-4','mi_kraskov_NN-4')

# classes = [['coupled map lattice'],['oscillator']]
# spis = ('bary-sq_dtw_mean','gc_gaussian_k-1_kt-1_l-1_lt-1')
# spis = ('tlmi_gaussian','lcss_constraint-sakoe-chiba')
# spis = ('ce_kozachenko','xme_kozachenko_k1')
# spis = ('cov-sq_EmpiricalCovariance','pdist_euclidean')
# spis = ('cov-sq_EmpiricalCovariance','dtw_constraint-sakoe-chiba')
# spis = ('mi_kraskov_NN-4','tlmi_kraskov_NN-4')
# spis = ('pdist_euclidean','cov-sq_EmpiricalCovariance')
# spis = ('tlmi_gaussian','dtw_constraint-sakoe-chiba')

# classes = [['var'],['oscillator']]
# spis = ('dtw_constraint-sakoe-chiba','pdist_euclidean')
# spis = ('psi_multitaper_mean_fs-1_fmin-0_fmax-0-5','phase_multitaper_mean_fs-1_fmin-0_fmax-0-5')
# spis = ('gpdcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5','sgc_nonparametric_max_fs-1_fmin-0-25_fmax-0-5')
# spis = ('mi_kraskov_NN-4_DCE','tlmi_kraskov_NN-4_DCE')
# spis = ('mi_kraskov_NN-4','tlmi_kraskov_NN-4')

# classes = [['var'],['coupled map lattice']]
# spis = ('mi_kraskov_NN-4','tlmi_kraskov_NN-4')
# spis = ('xme_kozachenko_k1','ce_kozachenko')

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise']
for d in datatypes:
    with open(os.path.join(resdir,d)+'.pkl','rb') as f:
        cf1 = cPickle.load(f)
        try:
            cf.merge(cf1)
        except NameError:
            cf = cf1


cmap = sns.color_palette()
all_classes = ['coupled map lattice','var','oscillator','cauchy','normal']
cmap_dict = {cls: cmap[i] for i, cls in enumerate(all_classes)}

# Load the calculator files based on classes
pkls = [cls_to_pkl[classes[0][0]],cls_to_pkl[classes[1][0]]]

d = {}
M = 6
N = 64
for pkl, cls in zip(pkls,classes):
    with open(os.path.join(resdir,pkl),'rb') as f:
        calc = cPickle.load(f)

        As = [calc.table[s].iloc[:M,:M] for s in spis]
        
        df = pd.concat(As,axis=1,keys=[s for s in spis]).stack()
        calc._rmmin()

        if calc.spis[spis[0]].issigned() and calc.spis[spis[1]].issigned():
            df[[s+' (modified)' for s in spis]] = pd.concat(As,axis=1,keys=[s for s in spis]).stack()
        else:
            df[[s+' (modified)' for s in spis]] = pd.concat(As,axis=1,keys=[s for s in spis]).stack().abs()
        df[[s+' (rank)' for s in spis]] = df[[s+' (modified)' for s in spis]].rank()
        d[cls[0]] = df

        for s in spis:
            A = As[0]
            linewidth = max(2 - A.shape[0] / 50, 0)

            fig, ax = plt.subplots(figsize=(10,10))
            ax.pcolormesh(A,cmap='binary',edgecolors='w',linewidth=linewidth)
            plt.tick_params(which='both', bottom=False,top=False,left=False,right=False,
                            labelbottom=False,labelleft=False)
            ax.invert_xaxis()
            fig.savefig(f'plots/{cls[0]}_{s}.jpg',dpi=300,bbox_inches='tight',pad_inches=0)

        dat = calc.dataset.to_numpy(squeeze=True)[:M,:N]

        fig, ax = plt.subplots(figsize=(12,6))
        ax.pcolormesh(dat,cmap=sns.color_palette('icefire_r', as_cmap=True),vmin=-2,vmax=2)
        plt.tick_params(which='both', bottom=False,top=False,left=False,right=False,
                                labelbottom=False,labelleft=False)
        ax.invert_xaxis()
        fig.savefig(f'plots/{cls[0]}.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df,x=spis[0],y=spis[1],s=75,color='k',ax=ax)
        fig.savefig(f'plots/{cls[0]}_{spis[0]}-{spis[1]}.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)

df = pd.concat(d.values(),axis=1,keys=d.keys())
df.columns.names = ['class','spi']
df = df.reorder_levels([1,0],axis=1).stack().reset_index()

plt.subplots()
sns.scatterplot(x=spis[0],y=spis[1],hue='class',data=df,palette=cmap_dict)

plt.subplots()
sns.scatterplot(x=spis[0]+' (modified)',y=spis[1]+' (modified)',hue='class',data=df,palette=cmap_dict)

plt.subplots()
sns.scatterplot(x=spis[0]+' (rank)',y=spis[1]+' (rank)',hue='class',data=df,palette=cmap_dict)

feature_matrix = cf.get_feature_matrix(sthresh=0.95,dthresh=0.05,dropduplicates=False)
feature_vector = feature_matrix.loc[spis]
feature_vector.name = ', '.join(spis)

cf.set_dgroups(classes=classes)

df = pd.DataFrame(feature_vector)
df['labels'] = cf.get_dgroup_names(feature_vector.index)

df.drop(index=df.index[df['labels']=='N/A'],inplace=True)

if flipped_axes:
    fig, ax = plt.subplots(figsize=(3,6))
    sns.histplot(data=df,y=feature_vector.name,hue='labels',palette=cmap_dict,bins=20,kde=False,stat='density',common_norm=False,element="step")
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
else:
    fig, ax = plt.subplots(figsize=(8,8))
    sns.histplot(data=df,x=feature_vector.name,hue='labels',palette=cmap_dict,bins=20,kde=False,stat='density',common_norm=False,element="step")
fig.savefig(f'plots/{classes[0]}-{classes[1]}_{spis[0]}-{spis[1]}.pdf',dpi=300,bbox_inches='tight',transparent=True)
plt.show()