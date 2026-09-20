import os
import _pickle as cPickle
import matplotlib.pyplot as plt
from utils import plot_stems
from scipy.stats import rankdata, spearmanr, pearsonr
import numpy as np
import statsmodels.api as sm

plt.rcParams.update({'font.size': 24})

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

# classes = [['coupled map lattice'],['oscillator']]
# spis = ('ce_kozachenko','xme_kozachenko_k1')
# spis = ('cov-sq_EmpiricalCovariance','pdist_euclidean')
# spis = ('cov-sq_EmpiricalCovariance','dtw_constraint-sakoe-chiba')
# spis = ('mi_kraskov_NN-4','tlmi_kraskov_NN-4')
# spis = ('pdist_euclidean','cov-sq_EmpiricalCovariance')
# spis = ('tlmi_gaussian','dtw_constraint-sakoe-chiba')

# classes = [['var'],['oscillator']]
# spis = ('dtw_constraint-sakoe-chiba','pdist_euclidean')
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

# Load the calculator files based on classes
pkls = [cls_to_pkl[classes[0][0]],cls_to_pkl[classes[1][0]]]

N = 64
marg = [lambda x : np.argmin(np.abs(x)),lambda x : np.argmax(np.abs(x))]

for pkl, cls, nparg in zip(pkls,classes,marg):
    print(f'Opening data from {pkl}')
    with open(os.path.join(resdir,pkl),'rb') as f:
        calc = cPickle.load(f)

        dat = calc.dataset.to_numpy(squeeze=True)[:,:N]

        rdiffs = []
        ms = []
        for i in range(dat.shape[0]):
            for j in range(i+1,dat.shape[0]):
                r, _ = pearsonr(dat[i],dat[j])
                r_s, _ = spearmanr(dat[i],dat[j])
                rdiffs.append(r-r_s)
                ms.append([i,j])
        Ms = ms[nparg(rdiffs)]

    mts = dat[Ms,:]
    r, _ = pearsonr(mts[0],mts[1])
    r_s, _ = spearmanr(mts[0],mts[1])
    print(f'r = {r}, r_s = {r_s} (r-r_s = {r-r_s})')

    # Plot the time-series data for the datasets
    x, y = mts[0,:], mts[1,:]
    r_x, r_y = rankdata(mts[0,:]), rankdata(mts[1,:])

    fig, _ = plot_stems(x)
    fig.savefig(f'plots/{cls[0]}_x-ts.pdf',dpi=300,bbox_inches='tight',transparent=True)
    fig, _ = plot_stems(y)
    fig.savefig(f'plots/{cls[0]}_y-ts.pdf',dpi=300,bbox_inches='tight',transparent=True)

    fig, ax = plt.subplots(figsize=(4,5))
    plt.scatter(x,y,c='k')
    olsfit = sm.OLS(y,sm.add_constant(x)).fit()
    x_s = np.array([x.min(),x.max()])
    plt.plot(x_s,x_s*olsfit.params[1] + olsfit.params[0],'r',linewidth=2,alpha=0.6,zorder=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.savefig(f'plots/{cls[0]}_xy.pdf',dpi=300,bbox_inches='tight',transparent=True)

    fig, ax = plt.subplots(figsize=(4,5))
    plt.scatter(r_x,r_y,c='k')
    olsfit = sm.OLS(r_y,sm.add_constant(r_x)).fit()
    rX_s = np.array([r_x.min(),r_x.max()])
    plt.plot(rX_s,rX_s*olsfit.params[1] + olsfit.params[0],'r',linewidth=2,alpha=0.6,zorder=2)
    ax.set_xlabel('rank(x)')
    ax.set_ylabel('rank(y)')
    fig.savefig(f'plots/{cls[0]}_rank-xy.pdf',dpi=300,bbox_inches='tight',transparent=True)
    plt.show()
