"""Post-hoc, development-selected explanation of existing centered PC1; no new fits promoted."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, rankdata
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.benchmark_latex_style import latex_style, prepare_latex_figure

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/order-parameter-inference/locking-preview-261007'
SOURCE = ROOT / 'results/order-parameter-inference/cross-frequency-locking-confirm-261006'


def main():
    bank = np.load(SOURCE / 'features.npz')
    rows = pd.read_csv(SOURCE / 'scores.csv')
    np.testing.assert_array_equal(bank['row_id'], rows.row_id)
    aa, bb = np.triu_indices(len(bank['spi_order']), 1)
    records, summaries = [], {}
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['CMU Serif', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'font.size': 7, 'axes.spines.top': True,
                         'axes.spines.right': True, 'legend.frameon': False})
    latex_style()
    for arm in ['random-noise', 'common-mode', 'internal-coupling']:
        select = rows.system.eq(arm).to_numpy()
        part = rows[select].reset_index(drop=True)
        raw = bank['z'][select]
        fit = part.role.eq('development').to_numpy()
        held = (~fit) & part.comparison_eligible.to_numpy()
        keep = np.isfinite(raw[fit]).all(0)
        keep[keep] &= raw[fit][:,keep].std(0) > 1e-10
        original = np.flatnonzero(keep)
        x = raw[:, keep].astype(float)
        x = np.where(np.isfinite(x), x, np.median(x[fit], axis=0))
        center = x[fit].mean(0)
        model = PCA(1, svd_solver='full').fit(x[fit] - center)
        w = model.components_[0]
        q = (x - center) @ w
        if np.corrcoef(q[fit], part.z_PC1[fit])[0, 1] < 0:
            w = -w
            q = -q
        error = float(abs(q - part.z_PC1).max())
        assert error < 1e-4, error
        scale = q[fit].std()
        low = fit & np.isclose(part.control, .08)
        high = fit & np.isclose(part.control, .12)
        delta = x[high].mean(0) - x[low].mean(0)
        contribution = w * delta / scale
        assert np.isclose(contribution.sum(), (q[high].mean()-q[low].mean())/scale)
        rank = np.argsort(-abs(contribution))
        # Fixed original development validity mask; leave-one-seed-out checks loading direction only.
        aligned = []
        for seed in sorted(part.seed[fit].unique()):
            train = fit & part.seed.ne(seed).to_numpy()
            other = PCA(1, svd_solver='full').fit(x[train]).components_[0]
            other *= 1 if other @ w >= 0 else -1
            aligned.append(other)
        aligned = np.array(aligned)
        nonlag = [j for j in rank if not any(str(bank['spi_order'][s]).startswith(('corr_', 'xcorr')) for s in [aa[original[j]],bb[original[j]]])]
        family_representatives, families = [], set()
        for j in nonlag:
            family=tuple(sorted(str(bank['spi_order'][s]).split('_')[0] for s in [aa[original[j]],bb[original[j]]]))
            if family not in families:
                family_representatives.append(j)
                families.add(family)
        for j in dict.fromkeys([*rank[:20], *nonlag[:40], *family_representatives[:20]]):
            k = original[j]
            records.append(dict(arm=arm,feature_index=int(k),spi_a=str(bank['spi_order'][aa[k]]),
                                spi_b=str(bank['spi_order'][bb[k]]),weight=w[j],
                                train_transition_contribution=contribution[j],nonlag=bool(j in nonlag),family_representative=bool(j in family_representatives),
                                same_sign_folds=int((aligned[:,j]*w[j]>0).sum()),
                                held_rho_Q=spearmanr(x[held,j], part.Q_lock[held]).statistic,
                                held_rho_nuisance=spearmanr(x[held,j], part.nuisance[held]).statistic))
        summaries[arm] = dict(score_replay_max_error=error, retained_features=len(w),
                              source_features_sha256=hashlib.sha256((SOURCE/'features.npz').read_bytes()).hexdigest(),
                              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              training_transition_score_change=float(contribution.sum()),
                              top4_abs_contribution_share=float(abs(contribution[rank[:4]]).sum()/abs(contribution).sum()),
                              min_leave_seed_loading_cosine=float((aligned @ w).min()))
        if arm != 'random-noise':
            continue
        fig, axes = plt.subplots(2,3,figsize=(8.4,6.4),constrained_layout=True,sharex=True,sharey=True)
        for ax in axes.flat:ax.set_box_aspect(1)
        titles = []
        for ax,j in zip(axes.flat,family_representatives[:6]):
            k=original[j]
            a,b=str(bank['spi_order'][aa[k]]),str(bank['spi_order'][bb[k]])
            group=pd.Series(x[held,j]).groupby(part.control[held].to_numpy());mean=group.mean()
            ax.plot(mean.index,mean,'s-',color='#0072B2',ms=2.7,lw=1.7)
            ax.fill_between(mean.index,group.quantile(.1),group.quantile(.9),color='#0072B2',alpha=.15,lw=0)
            ax.axvline(.1017,color='.6',ls=':',lw=.8)
            short=lambda s: s.replace('coint_johansen_max_eig_stat_order-1_ardiff-10','Johansen max-eigenvalue (lag 10)').replace('coint_johansen_trace_stat_order-1_ardiff-1','Johansen trace (lag 1)').replace('coint_johansen_trace_stat_order-1_ardiff-10','Johansen trace (lag 10)').replace('si_kernel_W-0.5_k-1','Kernel stochastic interaction').replace('te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1','Transfer entropy (KSG)').replace('lcss_constraint-itakura','LCSS (Itakura)').replace('dtw_constraint-sakoe-chiba_radius-auto','DTW (Sakoe–Chiba)').replace('dtw','DTW')
            ax.set(title=short(a)+'\n× '+short(b),ylim=(-1.05,1.05))
            ax.text(.04,.06,rf'$|\rho(z_{{ab}},Q)|={abs(spearmanr(x[held,j],part.Q_lock[held]).statistic):.2f}$',transform=ax.transAxes)
            titles.append([a,b])
        for ax in axes[-1]:ax.set_xlabel(r'Cross-coupling $\gamma$')
        for ax in axes[:,0]:ax.set_ylabel(r'SPI–SPI feature $z_{ab}$')
        fig.suptitle('No lagged-correlation endpoint: best development-ranked feature per identifier-family pair\n'
                     '$M=N=48$, $T=1000$ · varying sensor noise · held means and 10–90% instance bands',fontsize=10)
        prepare_latex_figure(fig)
        fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['png','svg']:fig.savefig(OUT/f'nonlag-features.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        summaries[arm]['nonlag_displayed_pairs']=titles
        fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.3), constrained_layout=True)
        for ax in axes.flat:ax.set_box_aspect(1)
        for ax, j in zip(axes, rank[:4]):
            k = original[j]
            a, b = str(bank['spi_order'][aa[k]]), str(bank['spi_order'][bb[k]])
            group = pd.Series(x[held, j]).groupby(part.control[held].to_numpy())
            mean = group.mean()
            ax.plot(mean.index, mean, 's-', color='#0072B2', ms=2.7, lw=1.7)
            ax.fill_between(mean.index, group.quantile(.1), group.quantile(.9), color='#0072B2', alpha=.15, lw=0)
            ax.axvline(.1017, color='.6', ls=':', lw=.8)
            ax.set(xlabel=r'Cross-coupling $\gamma$', title=f'{a}\n× {b}', ylim=(-1.05, 1.05))
            ax.title.set_fontsize(6.5)
            ax.text(.04,.05,rf'$|\rho(z_{{ab}},Q)|={abs(spearmanr(x[held,j],part.Q_lock[held]).statistic):.2f}$', transform=ax.transAxes, fontsize=7)
        axes[0].set_ylabel(r'SPI–SPI feature $z_{ab}$')
        fig.suptitle(r'Top four development-ranked contributions to the PC1 change from $\gamma=.08$ to $.12$'+'\n'
                     '$M=N=48$, $T=1000$ · varying sensor noise · held means and 10–90% instance bands', fontsize=10)
        prepare_latex_figure(fig)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        for ext in ['png','svg']:
            fig.savefig(OUT / f'individual-features.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        # Underlying edge profiles for the first selected feature; no costly SPI rerun.
        archive = np.load(ROOT/'data/order-parameter-inference/cross-frequency-locking-confirm-261006/observations.npz')
        pair_rows = []
        fig, axes = plt.subplots(1,3,figsize=(7.5,3.3),constrained_layout=True,sharex=True,sharey=True)
        for ax in axes.flat:ax.set_box_aspect(1)
        for ax,gamma in zip(axes,[.04,.10,.14]):
            candidates = part[held & np.isclose(part.control,gamma)]
            record = candidates.iloc[np.argmin(abs(candidates.eta.to_numpy()-.6))]
            observation = archive[record.row_id]
            def lag(tau,rank=False):
                stacked = np.vstack([observation[:,tau:],observation[:,:-tau]])
                if rank:
                    stacked=rankdata(stacked,axis=1)
                matrix=np.corrcoef(stacked)[:48,48:]
                return .5*(matrix+matrix.T)
            a,b=lag(2),lag(20,True)**2
            i,j=np.triu_indices(48,1)
            observed=float(np.corrcoef(a[i,j],b[i,j])[0,1])
            expected=float(raw[part.row_id.eq(record.row_id).to_numpy(),original[rank[0]]][0])
            assert abs(observed-expected)<1e-6,(observed,expected)
            for category,label,color in [(0,'within A','#0072B2'),(1,'within B','#D55E00'),(2,'within C','#009E73'),(3,'between communities','.6')]:
                mask=((i//16)==category)&((j//16)==category) if category<3 else (i//16)!=(j//16)
                ax.scatter(a[i[mask],j[mask]],b[i[mask],j[mask]],s=7,alpha=.4,color=color,label=label)
                pair_rows.append(dict(control=gamma,row_id=record.row_id,category=label,eta=record.eta,
                                      mean_a=float(a[i[mask],j[mask]].mean()),mean_b=float(b[i[mask],j[mask]].mean()),z=observed))
            ax.set(xlabel='Lag-2 Pearson',title=rf'$\gamma={gamma:.2f}$, $\eta={record.eta:.2f}$, $z_{{ab}}={observed:.2f}$')
        axes[0].set_ylabel('Squared lag-20 Spearman')
        axes[0].legend(fontsize=7)
        fig.suptitle(r'One dot per unordered channel pair; held recording nearest $\eta=.6$ at each fixed control'+'\n'
                     'Same feature as the first panel above; symmetric SPIs make ordered-pair correlation identical',fontsize=10)
        prepare_latex_figure(fig)
        fig.canvas.draw();fig.set_layout_engine('none')
        for ext in ['png','svg']:
            fig.savefig(OUT/f'edge-profiles.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        pd.DataFrame(pair_rows).to_csv(OUT/'edge-profiles.csv',index=False)
    pd.DataFrame(records).to_csv(OUT / 'feature-contributions.csv', index=False)
    (OUT / 'audit.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))
    print(pd.DataFrame(records).groupby('arm',sort=False).head(5).to_string(index=False))
    print(pd.DataFrame(records).query('arm == "random-noise" and nonlag').head(25).to_string(index=False))
    print(pd.DataFrame(records).query('arm == "random-noise" and family_representative').to_string(index=False))


def comparisons():
    plt.rcParams.update({'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],
                         'mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.frameon':False})
    latex_style()
    fig,axes=plt.subplots(2,3,figsize=(11.4,6.2),constrained_layout=True)
    clean=pd.read_csv(SOURCE.parent/'cross-frequency-locking-261006/scores.csv')
    noisy=pd.read_csv(SOURCE/'scores.csv');noisy=noisy[noisy.system.eq('random-noise')]
    for row,(data,title) in enumerate([(clean,'Clean, M=N=24'),(noisy,'Varying sensor noise, M=N=48')]):
        fit=data.role.eq('development');held=data.role.eq('evaluation')&data.comparison_eligible
        group=data[held].groupby('control').Q_lock
        for ax,(key,name) in zip(axes[row],[('z_PC1','SPI–SPI PC1 (centered)'),('mean_abs_r','Mean absolute Pearson'),('mean_PC1','Mean-SPI PC1')]):
            right=ax.twinx();right.spines['right'].set_visible(True)
            lines=ax.plot(group.mean().index,group.mean(),'o-',color='#222222',label='$Q$',ms=2.7)
            values=data[key] if key=='mean_abs_r' else (data[key]-data.loc[fit,key].mean())/data.loc[fit,key].std()
            g=values[held].groupby(data.control[held]);m=g.mean()
            lines+=right.plot(m.index,m,'s-',color='#0072B2',label='readout',ms=2.7)
            right.fill_between(m.index,g.quantile(.1),g.quantile(.9),color='#0072B2',alpha=.16,lw=0)
            ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Locking index $Q$',ylim=(-.05,1.05),title=title+'\n'+name)
            right.set(ylabel=r'$\langle |r|\rangle$' if key=='mean_abs_r' else '$q$ (fit SD)',ylim=(0,.45) if key=='mean_abs_r' else (-3,3))
            ax.axvline(.1017,color='.6',ls=':',lw=.8);ax.legend(lines,[l.get_label() for l in lines],fontsize=8,loc='upper left')
    fig.suptitle('Full p90 · T=1000 · held means; shaded readouts: 10–90% of instances',fontsize=10)
    prepare_latex_figure(fig)
    fig.canvas.draw();fig.set_layout_engine('none')
    for ext in ['png','svg']:fig.savefig(OUT/f'locking-overview.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Exact matrices copied read-only from the existing farm, verified against its receipt.
    rows=pd.read_csv(SOURCE/'scores.csv');bank=np.load(SOURCE/'features.npz')
    sources=json.loads((SOURCE/'sources.json').read_text())['sources']
    source_by_id={r['row_id']:r for r in sources}
    aa,bb=np.triu_indices(len(bank['spi_order']),1);order=list(bank['spi_order'])
    pairs=[('lcss_constraint-itakura','te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'),
           ('cov_EmpiricalCovariance','te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1')]
    fig,axes=plt.subplots(2,3,figsize=(8.4,6.4),constrained_layout=True,sharex='row',sharey='row')
    for ax in axes.flat:
        ax.set_box_aspect(1)
        for spine in ax.spines.values():spine.set_visible(True)
        ax.tick_params(labelsize=7)
    output=[]
    for col,(suffix,gamma) in enumerate([('g040-s310',.04),('g100-s310',.10),('g140-s309',.14)]):
        path=OUT/'mpi-examples'/f'{suffix}.npz';row_id='xfreq21-random-noise-'+suffix
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_by_id[row_id]['mpi_sha256']
        record=rows[rows.row_id.eq(row_id)].iloc[0];mpis=np.load(path)
        i,j=np.where(~np.eye(48,dtype=bool))
        for row,(a,b) in enumerate(pairs):
            av,bv=mpis[a][i,j],mpis[b][i,j]
            finite=np.isfinite(av)&np.isfinite(bv);assert finite.all()
            z=float(np.corrcoef(av,bv)[0,1]);ia,ib=sorted([order.index(a),order.index(b)])
            k=int(np.flatnonzero((aa==ia)&(bb==ib))[0]);idx=int(rows.index[rows.row_id.eq(row_id)][0])
            assert abs(z-float(bank['z'][idx,k]))<1e-6
            for category,label,color in [(0,'within A','#0072B2'),(1,'within B','#D55E00'),(2,'within C','#009E73'),(3,'A–B','#CC79A7'),(4,'A/B–C','.65')]:
                mask=((i//16)==category)&((j//16)==category) if category<3 else (((i//16)==0)&((j//16)==1)|((i//16)==1)&((j//16)==0)) if category==3 else ((i//16)==2)^((j//16)==2)
                axes[row,col].scatter(av[mask],bv[mask],s=5,alpha=.3,color=color,label=label)
                output.append(dict(row_id=row_id,spi_a=a,spi_b=b,category=label,mean_a=float(av[mask].mean()),mean_b=float(bv[mask].mean()),z=z))
            axes[row,col].set(title=rf'$\gamma={gamma:.2f}$, $\eta={record.eta:.2f}$, $z_{{ab}}={z:.2f}$',xlabel='LCSS similarity (Itakura)' if row==0 else 'Empirical covariance (standardized channels)')
            if col==0:axes[row,col].set_ylabel('Transfer entropy (KSG)')
    axes[0,0].legend(fontsize=7,loc='upper left')
    fig.suptitle('Underlying SPI entries · one point per directed channel pair\n'+r'Held records nearest $\eta=.6$ at fixed controls; exact cached matrices',fontsize=10)
    prepare_latex_figure(fig)
    fig.canvas.draw();fig.set_layout_engine('none')
    for ext in ['png','svg']:fig.savefig(OUT/f'nonlag-edge-profiles.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(output).to_csv(OUT/'nonlag-edge-profiles.csv',index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        if '--figures-only' not in sys.argv:
            main()
        comparisons()
