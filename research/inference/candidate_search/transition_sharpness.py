"""Descriptive, affine-invariant change concentration; not a fitted phase law."""
import numpy as np
import pandas as pd


def curve_shape(control,values):
    x=np.asarray(control,float);y=np.asarray(values,float)
    movement=np.abs(np.diff(y));total=movement.sum()
    if not np.isfinite(y).all() or total<=1e-12:
        return dict(monotonicity=np.nan,width=np.nan,midpoint=np.nan)
    monotonicity=abs(y[-1]-y[0])/total
    cumulative=np.r_[0,np.cumsum(movement)]/total
    # Uniformly distribute each measured change over its observed control interval.
    x10,x50,x90=np.interp([.1,.5,.9],cumulative,x)
    return dict(monotonicity=float(monotonicity),width=float(x90-x10),midpoint=float(x50))


def summarize(scores,fields,output,repeats=1000):
    """Resample whole held seeds; guard against small contrast and reversals."""
    records=[]
    for system,part in scores[scores.role=='evaluation'].groupby('system',sort=False):
        seeds=sorted(part.seed.unique());controls=np.sort(part.control.unique())
        rng=np.random.default_rng(261055)
        draws=rng.integers(len(seeds),size=(repeats,len(seeds)))
        truth=part.groupby('control').CLE.mean().reindex(controls).to_numpy()
        crossings=np.flatnonzero(truth[:-1]*truth[1:]<=0)
        boundary=np.nan
        if len(crossings)==1:
            j=crossings[0];boundary=controls[j]-truth[j]*(controls[j+1]-controls[j])/(truth[j+1]-truth[j])
        for field in fields:
            grid=part.pivot(index='seed',columns='control',values=field).reindex(index=seeds,columns=controls).to_numpy()
            shape=curve_shape(controls,np.nanmean(grid,axis=0))
            boot=np.nanmean(grid[draws],axis=1)
            widths=np.array([curve_shape(controls,b)['width'] for b in boot])
            contrast=boot[:,-2:].mean(1)-boot[:,:2].mean(1)
            lo,hi=np.quantile(contrast,[.025,.975])
            eligible=bool((lo>0 or hi<0) and shape['monotonicity']>=.8)
            records.append(dict(system=system,method=field,**shape,
                min_instances_per_control=int(np.isfinite(grid).sum(0).min()),
                width_low=float(np.nanquantile(widths,.025)),width_high=float(np.nanquantile(widths,.975)),
                contrast_low=float(lo),contrast_high=float(hi),single_transition_interpretation=eligible,
                physical_CLE_zero=float(boundary),midpoint_distance=float(abs(shape['midpoint']-boundary))))
    frame=pd.DataFrame(records);frame.to_csv(output,index=False);return frame
