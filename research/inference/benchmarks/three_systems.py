"""Presentation-only comparison of frozen SPI–SPI inference studies.

No extraction or outcome-based refitting. Original scores remain in their own
units; the independently fitted representations are never pooled.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mtsspi.corpus_visualization import scale_timeseries
from research.inference.locking.plot_locking_composites_261008 import choose_records
from research.empirical.hardware_pilot.calibrate_hardware_pilot import phase_properties

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/order-parameter-inference/three-systems-261015'
Q_COLOR, Z_COLOR = '#222222', '#26828e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    """Portable notebook typography; no external TeX executables required."""
    plt.rcParams.update({'text.usetex':False, 'mathtext.fontset':'cm',
        'font.family':'serif', 'font.serif':['DejaVu Serif'],
        'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10, 'legend.fontsize':7,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.spines.top':False,
        'axes.spines.right':False, 'legend.frameon':False, 'figure.dpi':180,
        'lines.linewidth':1.7, 'lines.markersize':2.7,
        'xtick.direction':'out', 'ytick.direction':'out'})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=180 if ext=='png' else 600, bbox_inches='tight')
    plt.close(fig)
    return OUT/f'{name}.png'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_cml():
    pilot=ROOT/'data/order-parameter-inference/cml2d/cml2d_period_doubling_260911'
    confirm=ROOT/'data/order-parameter-inference/cml2d/cml2d_confirmation_260911/primary-analysis'
    report=json.loads((confirm/'confirmation-report.json').read_text())
    check(report['gate']['passes'], 'CML confirmation gate failed')
    for filename,digest in report['frozen_hashes'].items():
        check(sha(pilot/'primary-analysis'/filename)==digest, f'CML frozen {filename} changed')
    frame=pd.read_csv(confirm/'scores.csv')
    check(len(frame)==544 and frame.eligible.all() and frame.seed.nunique()==32, 'CML scope changed')
    with np.load(confirm/'model.npz') as m, np.load(confirm/'features.npz') as f:
        check(f['row_id'].tolist()==frame.row_id.tolist(), 'CML feature row ordering changed')
        selected=f['z'][:,m['keep']]
        replay=(np.where(np.isfinite(selected),selected,m['impute'])-m['center'])@m['component']/m['score_scale']
        error=float(np.max(abs(replay-frame.q)))
        check(error<1e-7, 'CML score replay failed')
    manifest=json.loads((pilot/'primary/manifest.json').read_text())
    examples=[]
    with np.load(pilot/'primary/observations.npz') as obs:
        for r in (3.84,3.86212,3.89):
            row=next(rw for rw in manifest['rows'] if rw['seed']==26091115 and rw['r']==r)
            master=pilot/'primary-masters'/row['master']
            check(sha(master)==row['master_sha256'], 'CML illustrative master changed')
            with np.load(master) as a:
                x=obs[row['row_id']]
                np.testing.assert_array_equal(x,a['observed'][:1000,0,:32].T)
                examples.append(dict(control=r,x=x,mean=a['global_mean'][:100],field=a['final_state'],
                    sensors=a['sensor_indices'][0,:32],row_id=row['row_id']))
    baseline=pd.read_csv(ROOT/'results/baseline-comparison_260921/inference-CML2D.csv').set_index('row_id').loc[frame.row_id]
    np.testing.assert_allclose(baseline.q,frame.q,atol=1e-7)
    np.testing.assert_allclose(baseline.Q_reference,frame.Q_reference,atol=1e-7)
    check(baseline.common_eligible.all(), 'CML baseline eligibility changed')
    frame['mean_PC1']=baseline.mean_PC1.to_numpy()
    return dict(frame=frame,examples=examples,report=report,replay_error=error,
        score_sha256=sha(confirm/'scores.csv'),rho=float(spearmanr(frame.q,frame.Q_reference).statistic))


def load_locking():
    run='cross-frequency-locking-snr-261006'
    result=ROOT/'results/order-parameter-inference'/run
    frame=pd.read_csv(result/'scores.csv').query("system == 'random-noise'").copy()
    check(len(frame)==336 and frame.comparison_eligible.all(), 'Locking scope changed')
    check((frame.role=='evaluation').sum()==168, 'Locking held set changed')
    metadata=json.loads((result/'analysis.json').read_text())
    check(sha(result/'features.npz')==metadata['features_sha256'], 'Locking feature bank changed')
    audit=json.loads((ROOT/'results/order-parameter-inference/locking-composites-261008/audit.json').read_text())['random-noise-m24']
    check(sha(result/'scores.csv')==audit['scores_sha256'], 'Locking frozen scores changed')
    chosen=choose_records(frame,'eta',.6)
    examples=[]
    with np.load(ROOT/'data/order-parameter-inference'/run/'observations.npz') as obs:
        for row,old in zip(chosen,audit['records'],strict=True):
            x=obs[row.row_id]
            check(row.row_id==old['row_id'] and hashlib.sha256(x.tobytes()).hexdigest()==old['array_sha256'],
                  'Locking snapshot selection/input changed')
            examples.append(dict(control=row.control,eta=row.eta,x=x,row_id=row.row_id))
    held=frame.query("role == 'evaluation'")
    metrics={key:float(abs(spearmanr(held[key],held.Q_lock).statistic)) for key in ['z_PC1','mean_abs_r','mean_PC1']}
    return dict(frame=frame,examples=examples,metrics=metrics,score_sha256=sha(result/'scores.csv'))


def load_hardware():
    from research.empirical.hardware_pilot.full_p90_analysis import load_run
    run=ROOT/'results/hardware-pilot-261012/full-p90-r9-261014'
    bank=load_run(run)  # Source, catalogue, full-edge, version and completeness audit.
    frame=pd.read_csv(run/'analysis/scores.csv')
    report=json.loads((run/'analysis/summary.json').read_text())
    check(len(frame)==45 and not frame.er_missing_fraction.any(), 'Hardware scope/availability changed')
    with np.load(run/'analysis/features.npz') as f:
        np.testing.assert_array_equal(f['z'],bank.z)
        scores=(bank.z[:,f['er_features']].astype(float)-f['er_center'])@f['er_components'].T
        np.testing.assert_allclose(scores,frame[['er_pc1','er_pc2']],atol=1e-8)
        check(np.array_equal(f['er_fit_indices'],np.flatnonzero(frame.topology.eq('ER'))), 'Hardware fit used non-ER records')
    physical=[]; examples=[]
    with np.load(run/'observations.npz') as obs:
        for row in bank.rows:
            x=obs[row['name']]; p=phase_properties(x,(90,140))
            physical.append(dict(name=row['name'],frequency_range=float(np.ptp(p['frequency'])),mean_R=p['R1']))
            if row['topology']=='ER' and row['step'] in (21,23,26):
                # Reference channel is fixed; phase traces are display diagnostics only.
                relative=np.unwrap(np.angle(p['unit']*p['unit'][0].conj()),axis=1)/(2*np.pi)
                relative-=relative[:,0,None]
                examples.append(dict(control=row['voltage'],step=row['step'],x=x,phase=relative,row_id=row['name']))
    frame=frame.merge(pd.DataFrame(physical),on='name',validate='one_to_one')
    check(len(examples)==3, 'Missing hardware illustrative recordings')
    intervals={}
    for topology,before,after in [('ER',23,24),('SW',31,34)]:
        part=frame.query('topology == @topology').sort_values('step')
        i=int(np.argmax(abs(np.diff(part.er_pc1))))
        intervals[topology]=[int(part.iloc[i].step),int(part.iloc[i+1].step)]
        check(intervals[topology]==[before,after], 'Hardware largest-step result changed')
    return dict(frame=frame,examples=examples,report=report,intervals=intervals,
                score_sha256=sha(run/'analysis/scores.csv'),bank=bank)


def draw_overview(cml):
    fig=plt.figure(figsize=(10,5.1),layout='constrained')
    grid=fig.add_gridspec(2,3,height_ratios=[2,1])
    captions=[('2D coupled maps', 'Change local nonlinearity $r$', 'Chaotic sites; collective period-two order', '32 sensors from 65,536 sites; T=1,000'),
              ('Resonant communities', r'Change cross-coupling $\gamma$', 'Slow A locks to fast B at 2:1; C is a reference', '24 observed oscillators; T=1,000; varying noise'),
              ('Measured electronic oscillators', 'Change global coupling-control voltage', 'Fixed wiring; recruitment into frequency locking', '28 channels; T=5,000; full 10 s')]
    for col,(title,control,question,shape) in enumerate(captions):
        ax=fig.add_subplot(grid[0,col]); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        if col==0:
            ax.imshow(cml['examples'][0]['field'],extent=(.12,.88,.35,.93),origin='lower',cmap='viridis',vmin=0,vmax=1,aspect='auto')
            sensors=cml['examples'][0]['sensors'];side=256
            ax.scatter(.12+.76*(sensors%side)/side,.35+.58*(sensors//side)/side,s=9,facecolors='none',edgecolors='white',lw=.6)
        else:
            if col==1:
                for j,(label,color) in enumerate(zip('ABC',['#31688e','#d95f02','#777777'])):
                    center=(.18+j*.32,.63);theta=np.linspace(0,2*np.pi,8,endpoint=False)
                    ax.scatter(center[0]+.095*np.cos(theta),center[1]+.12*np.sin(theta),s=24,color=color)
                    ax.text(center[0],.39,label,ha='center')
                ax.add_patch(FancyArrowPatch((.29,.65),(.39,.65),arrowstyle='<->',mutation_scale=12,color='.3'))
                ax.text(.34,.84,'2:1',ha='center')
            else:
                theta=np.linspace(0,2*np.pi,28,endpoint=False)
                ax.scatter(.5+.32*np.cos(theta),.64+.25*np.sin(theta),s=18,color=Z_COLOR)
                ax.text(.5,.64,'28 physical\noscillators',ha='center',va='center')
                ax.text(.5,.34,'Schematic; edges omitted',ha='center',fontsize=7)
        ax.set_title(title);ax.text(.5,.24,control,ha='center',fontsize=9)
        ax.text(.5,.13,question,ha='center',fontsize=7.2);ax.text(.5,.02,shape,ha='center',fontsize=7.2)
    ax=fig.add_subplot(grid[1,:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    blocks=[('One recording',r'$X: M\times T$'),('Dependence profiles',r'$A^{(1)},\ldots,A^{(K)}$'),
            ('Compare the profiles',r'$z_{ab}=\mathrm{corr}_{i\ne j}(A^{(a)}_{ij},A^{(b)}_{ij})$'),('Across recordings',r'Frozen PC1 coordinate $q$')]
    for i,(label,formula) in enumerate(blocks):
        left=.005+i*.25
        ax.add_patch(Rectangle((left,.25),.23,.57,facecolor='#f2f5f5',edgecolor='.7',lw=.6))
        ax.text(left+.115,.66,label,ha='center',fontsize=9)
        ax.text(left+.115,.43,formula,ha='center',fontsize=8)
        if i<3:ax.annotate('',xy=(left+.25,.53),xytext=(left+.23,.53),arrowprops={'arrowstyle':'->','lw':.8})
    ax.text(.5,.04,'Fit without the physical reference; then compare q with independently defined physics or measured diagnostics.',ha='center',fontsize=8)
    return save(fig,'what-we-test')


def heatmaps(fig, axes, examples, *, count, dt, start=0, communities=False):
    scaled=[scale_timeseries(e['x'],'robust')[0] for e in examples]
    limit=max(float(np.quantile(abs(x),.99)) for x in scaled)
    from seaborn import color_palette
    cmap=color_palette('icefire',as_cmap=True)
    for ax,x in zip(axes,scaled,strict=True):
        m=x.shape[0];image=ax.imshow(x[:,start:start+count],origin='lower',aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,
            extent=(start*dt,(start+count)*dt,.5,m+.5),interpolation='nearest',rasterized=True)
        ax.set(xlabel='Time (s)' if dt==.002 else ('Time (model units)' if dt==.5 else 'Consecutive step'),ylabel='Observed channel',yticks=[1,m])
        if communities:
            for y in (8.5,16.5):ax.axhline(y,color='.6',lw=.5)
            ax.set(yticks=[4.5,12.5,20.5],yticklabels=['A','B','C'],ylabel='Community')
    fig.colorbar(image,ax=list(axes),shrink=.7,label='Robust-scaled value',extend='both')


def curve(frame, control, key):
    g=frame.groupby(control)[key]
    return g.mean(),g.quantile(.1),g.quantile(.9)


def dual(ax, frame, control, truth, score, *, xlabel, truthlabel, scorelabel='Frozen $q$', log=False, bands=True):
    right=ax.twinx();right.spines['right'].set_visible(True)
    for axis,key,color,marker,label in [(ax,truth,Q_COLOR,'o',truthlabel),(right,score,Z_COLOR,'s',scorelabel)]:
        mean,lo,hi=curve(frame,control,key)
        axis.plot(mean.index,mean,marker+'-',color=color,ms=2.7,lw=1.4,label=label)
        if bands:axis.fill_between(mean.index,lo,hi,color=color,alpha=.14,lw=0)
        axis.set_ylabel(label,color=color);axis.tick_params(axis='y',labelcolor=color)
    ax.set_xlabel(xlabel)
    if log:ax.set_yscale('log')
    return right


def draw_cml(data):
    fig,axes=plt.subplots(4,3,figsize=(10,9.6),layout='constrained',gridspec_kw={'height_ratios':[.8,1,1.05,1.15]})
    for ax,e in zip(axes[0],data['examples'],strict=True):
        ax.plot(np.arange(100),e['mean'],color=Q_COLOR,lw=.8)
        ax.set(title=rf"$r={e['control']:g}$",ylabel=r'Global $\bar{x}(t)$',xlabel='Consecutive step',ylim=(.3,.9))
    heatmaps(fig,axes[1],data['examples'],count=100,dt=1)
    for ax,e in zip(axes[2],data['examples'],strict=True):
        image=ax.imshow(e['field'],origin='lower',cmap='viridis',vmin=0,vmax=1,
                        extent=(-.5,255.5,-.5,255.5),interpolation='nearest',rasterized=True)
        sensors=e['sensors']
        ax.scatter(sensors%256,sensors//256,s=5,facecolors='none',edgecolors='white',lw=.35)
        ax.set(xlabel='Lattice column',ylabel='Lattice row',xticks=[0,128,255],yticks=[0,128,255])
        ax.set_box_aspect(1)
    fig.colorbar(image,ax=list(axes[2]),shrink=.7,label='Final lattice state $x$')
    f=data['frame']
    dual(axes[3,0],f,'r','Q_reference','q',xlabel='Local nonlinearity $r$',truthlabel='Period-two $Q$')
    axes[3,0].set_title('SPI–SPI PC1: fresh seeds')
    axes[3,0].axvline(3.86212,color='.5',lw=.8,ls=':')
    dual(axes[3,1],f,'r','Q_reference','mean_PC1',xlabel='Local nonlinearity $r$',truthlabel='Period-two $Q$',scorelabel='Mean-MPI PC1')
    axes[3,1].set_title('Mean-MPI PC1')
    dual(axes[3,2],f,'r','Q_reference','mean_abs_correlation',xlabel='Local nonlinearity $r$',truthlabel='Period-two $Q$',scorelabel=r'Mean $|r_{ij}|$')
    axes[3,2].set_title('Strength also tracks this transition')
    return save(fig,'cml2d')


def draw_locking(data):
    fig,axes=plt.subplots(3,3,figsize=(10,7.1),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.15]})
    bound=max(float(abs(e['x'][[0,8]]).max()) for e in data['examples'])*1.04
    for ax,e in zip(axes[0],data['examples'],strict=True):
        ax.scatter(e['x'][0],e['x'][8],s=2,alpha=.2,color=Z_COLOR,linewidths=0,rasterized=True)
        ax.set(title=rf"$\gamma={e['control']:.2f},\ \eta={e['eta']:.2f}$",xlabel='Slow A signal (SD)',ylabel='Fast B signal (SD)',xlim=(-bound,bound),ylim=(-bound,bound))
        ax.set_box_aspect(1)
    heatmaps(fig,axes[1],data['examples'],count=200,dt=.5,communities=True)
    held=data['frame'].query("role == 'evaluation'")
    for ax,key,label in zip(axes[2],['z_PC1','mean_PC1','mean_abs_r'],['SPI–SPI PC1','Mean-MPI PC1','Mean absolute Pearson'],strict=True):
        dual(ax,held,'control','Q_lock',key,xlabel=r'Cross-coupling $\gamma$',truthlabel='Harmonic locking $Q$',scorelabel=r'Mean $|r_{ij}|$' if key=='mean_abs_r' else 'Frozen $q$')
        ax.set_ylim(-.06,1.06);ax.axvline(.1017,color='.5',lw=.8,ls=':')
        ax.set_title(label+'\n'+rf"$|\rho_s|={data['metrics'][key]:.3f}$")
    return save(fig,'noisy-locking')


def draw_hardware(data):
    fig,axes=plt.subplots(3,3,figsize=(10,7.2),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.2]})
    phase_lo=min(float(e['phase'].min()) for e in data['examples'])
    phase_hi=max(float(e['phase'].max()) for e in data['examples'])
    phase_pad=.05*max(phase_hi-phase_lo,1.)
    for ax,e in zip(axes[0],data['examples'],strict=True):
        phase=e['phase'];t=.5+np.arange(phase.shape[1])/500
        for row in phase:ax.plot(t[::5],row[::5],color=Z_COLOR,lw=.55,alpha=.45)
        ax.set(title=f"ER setting {e['step']} ({e['control']:.3f} V)",xlabel='Time (s)',ylabel='Relative phase (cycles)',ylim=(phase_lo-phase_pad,phase_hi+phase_pad))
    heatmaps(fig,axes[1],data['examples'],count=100,dt=.002,start=500)
    for ax,top,bounds in zip(axes[2,:2],['ER','SW'],[(23,24),(31,34)],strict=True):
        frame=data['frame'].query('topology == @top and step <= 87')
        dual(ax,frame,'voltage','frequency_range','er_pc1',xlabel='Coupling-control voltage (V)',truthlabel='Carrier frequency range (Hz)',log=True,bands=False)
        ax.axvspan(bounds[0]*.005,bounds[1]*.005,color='.5',alpha=.13,lw=0)
        ax.set_title('ER: representation fit' if top=='ER' else 'Small-world: frozen projection')
    for top,color in [('ER',Z_COLOR),('SW','#d95f02')]:
        frame=data['frame'].query('topology == @top')
        axes[2,2].plot(frame.voltage,frame.er_pc1,'s-',color=color,label=top,ms=3,lw=1)
    axes[2,2].set(xlabel='Coupling-control voltage (V)',ylabel='Same ER-fitted $q$',title='Entire selected control range')
    axes[2,2].legend()
    return save(fig,'hardware')


def draw_hardware_edges(data):
    from research.empirical.hardware_pilot.full_p90_analysis import carrier_groups, plot_edge_contrast
    bank=data['bank'];names={(r['topology'],r['step']):i for i,r in enumerate(bank.rows)}
    feature=next(i for i,s in enumerate(bank.schema) if {s.spi_a,s.spi_b}=={'si_kernel_W-0.5_k-1','ids'})
    rows=[names[k] for k in [('ER',23),('ER',24),('SW',31),('SW',34)]]
    fig=plot_edge_contrast(bank,feature,rows,carrier_groups(bank))
    return save(fig,'hardware-edge-explanation')


def write_receipt(cml,locking,hardware):
    rows=[dict(system='CML2D',observed='M32 / N65536',samples=1000,fit='4 development seeds × 9 controls',evaluation='32 new seeds × 17 controls',reference='Disjoint future period-two Q',z_rho=cml['rho']),
          dict(system='Noisy 2:1 locking',observed='M=N=24',samples=1000,fit='8 development seeds × 21 controls',evaluation='8 held seeds × 21 controls',reference='Disjoint future harmonic-locking Q',z_rho=locking['metrics']['z_PC1']),
          dict(system='Hardware',observed='M=N=28',samples=5000,fit='29 ER recordings',evaluation='16 small-world recordings',reference='Measured carrier dispersion; no known universal Q',z_rho=None)]
    pd.DataFrame(rows).to_csv(OUT/'study-summary.csv',index=False)
    record=dict(studies=rows,source_score_hashes={k:v['score_sha256'] for k,v in [('cml',cml),('locking',locking),('hardware',hardware)]},
        cml_replay_max_error=cml['replay_error'],hardware_records=45,hardware_version=hardware['report']['computation'],
        hardware_coordinate_replay='validated original r9 z and ER projection',hardware_largest_adjacent_jumps=hardware['intervals'],
        locking_snapshot_ids=[e['row_id'] for e in locking['examples']],cml_snapshot_ids=[e['row_id'] for e in cml['examples']],
        hardware_snapshot_ids=[e['row_id'] for e in hardware['examples']],
        fitting='No fitting in this presentation; independent frozen models, no pooling across systems or versions.',
        plotting='Shared full-record robust heatmap scaling within each system; display crops only. Curves retain original units/signs; dual axes independently ranged. Synthetic bands are 10–90% seed-instance ranges, not confidence intervals. No hardware uncertainty bands.',
        source_sha256=sha(__file__))
    record['feature_gallery_source_sha256']=sha(Path(__file__).with_name('coordinate_features.py'))
    record['lattice']='Archived final state after the future reference window, same fixed held seed/control as the observed portraits; white circles show sensor locations, not simultaneous observation snapshots.'
    record['galleries']='Original PC1 coefficients; filtered views exclude two classical-correlation endpoints and duplicate displayed curves (atol 1e-7), without refitting. Edges replay archived ordered profiles.'
    (OUT/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    return pd.DataFrame(rows)
