"""Record-standardized ordered SPI edges and a small invariant learned readout."""
import numpy as np
import torch
from torch import nn


def standardize_edges(mpis, order):
    """Same invalid-column contract as Pearson z; no fitted population transform."""
    m=np.asarray(mpis[order[0]]).shape[0]
    mask=~np.eye(m,dtype=bool)
    vectors=np.stack([np.asarray(mpis[k],dtype=np.float64)[mask] for k in order],axis=1)
    finite=np.isfinite(vectors).all(axis=0)
    centered=np.zeros_like(vectors)
    centered[:,finite]=vectors[:,finite]-vectors[:,finite].mean(axis=0)
    norm=np.linalg.norm(centered,axis=0)
    valid=finite & np.isfinite(norm) & (norm>=1e-12)
    normalized=np.zeros_like(vectors)
    normalized[:,valid]=centered[:,valid]*(np.sqrt(len(vectors))/norm[valid])
    return normalized.astype(np.float32),valid


def pack_inputs(values,validity):
    """B x E x K values plus B x K validity, with no padded edges accepted."""
    assert values.ndim==3 and validity.shape==(len(values),values.shape[-1])
    return np.concatenate([values,np.broadcast_to(validity[:,None,:],values.shape)],axis=-1).astype(np.float32)


class SPIEdgePool(nn.Module):
    def __init__(self,spec):
        super().__init__();w=spec['edge_width'];h=spec['head_width']
        self.edge=nn.Sequential(nn.Linear(spec['input_width'],w),nn.GELU(),nn.Linear(w,w),nn.GELU())
        self.head=nn.Sequential(nn.Linear(2*w,h),nn.GELU(),nn.Linear(h,1))

    def forward(self,x):
        features=self.edge(x)
        pooled=torch.cat([features.mean(1),torch.sqrt(features.var(1,unbiased=False)+1e-6)],dim=1)
        return self.head(pooled).squeeze(-1)
