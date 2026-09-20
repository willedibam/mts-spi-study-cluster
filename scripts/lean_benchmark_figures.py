"""Presentation-only composites; never fit or alter an inference model."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.corpus_visualization import plot_mts_heatmap, scale_timeseries


def snapshots(root, system, controls):
    result=[]
    for c in controls:
        with np.load(root/f'data/order_parameter/lean_snapshots_260916/{system}-{c:g}.npz') as a:
            result.append({key:a[key].copy() for key in a.files})
        if system == 'stuart-landau':
            # The generator stores complex Z(t), not its magnitude.
            result[-1]['trace'] = np.abs(result[-1]['trace'])
    return result


def composite(examples, controls, symbol, tracking, *, title, output, lattice=False,
              field_label='Full system: space-time', trace_label=None, row_labels=None,
              robust_limit=None):
    """One fixed record at each control; final row is an across-control ensemble."""
    has_trace=trace_label is not None and all('trace' in a for a in examples)
    has_field=all('field' in a for a in examples)
    heights=([.8] if has_trace else [])+[1.5]+([2.1] if has_field else [])+[2.4]
    fig=plt.figure(figsize=(10.2, sum(heights)*1.13),constrained_layout=True)
    grid=fig.add_gridspec(len(heights),3,height_ratios=heights)
    limit=max(scale_timeseries(a['X'],'robust')[1] for a in examples) if robust_limit is None else robust_limit
    heat_axes=[];field_axes=[]
    for column,(a,c) in enumerate(zip(examples,controls,strict=True)):
        row=0
        anchor=['Range start','Boundary neighbourhood','Range end'][column]
        if has_trace:
            ax=fig.add_subplot(grid[row,column]);row+=1
            ax.plot(np.arange(100),a['trace'][:100],color='#222222',lw=.8)
            ax.set(ylabel=trace_label,title=rf'{anchor}: ${symbol}={c:g}$',xlim=(0,99))
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelbottom=False)
        ax=fig.add_subplot(grid[row,column]);row+=1
        plot_mts_heatmap(a['X'],method='robust',ax=ax,colorbar=False)
        ax.images[0].set_clim(-limit,limit)
        ax.set(xlim=(-.5,99.5),xlabel='Consecutive sample',ylabel='Observed channel')
        if not has_trace:
            ax.set_title(rf'{anchor}: ${symbol}={c:g}$')
        if row_labels is not None:
            ax.set_yticks(np.arange(len(row_labels)),row_labels)
        ax.spines[['top','right']].set_visible(False)
        heat_axes.append(ax)
        if has_field:
            ax=fig.add_subplot(grid[row,column]);field_axes.append(ax)
            im=ax.imshow(a['field'],origin='lower',aspect='equal' if lattice else 'auto',
                cmap='coolwarm',vmin=-1,vmax=1,interpolation='nearest',rasterized=True)
            ax.set(title=field_label,xlabel='Lattice column' if lattice else 'Consecutive sample',
                ylabel='Lattice row' if lattice else 'Physical site')
            if lattice:
                ij=a['sensors']
                ax.scatter(ij[:,1],ij[:,0],s=10,facecolors='none',edgecolors='white',linewidths=.65)
            else:
                ax.scatter(np.full(len(a['sensors']),-2),a['sensors'],marker='>',s=6,
                    color='black',clip_on=False)
            ax.spines[['top','right']].set_visible(False)
    fig.colorbar(heat_axes[-1].images[0],ax=heat_axes,shrink=.7,
        label='Robust-scaled value',extend='neither' if robust_limit is None else 'both')
    if has_field:
        fig.colorbar(im,ax=field_axes,shrink=.7,label=r'Raw state $x$')
    tracking(fig.add_subplot(grid[-1,:]))
    fig.suptitle(title)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(output.with_suffix('.'+suffix),dpi=300,bbox_inches='tight')
    return fig


def corpus_examples(corpus, controls, seed):
    """Exact stored inference inputs, selected by declared control and seed only."""
    rows=json.loads((corpus/'manifest.json').read_text())['rows']
    out=[]
    with np.load(corpus/'observations.npz',allow_pickle=False) as a:
        for c in controls:
            matches=[r for r in rows if r['seed']==seed and np.isclose(r['control'],c,atol=1e-10,rtol=0)]
            assert len(matches)==1
            out.append(dict(X=a[matches[0]['row_id']].copy()))
    return out
