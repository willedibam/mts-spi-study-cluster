"""Grouped binary neural fitting; no target dataset is accepted during training."""
import time
import numpy as np
import torch
from torch import nn
from src.representation_state_neural import make_encoder,seed_torch


def date_state_weights(archives,y):
    """Equal mass per labelled date and state despite unequal interval lengths."""
    weights=np.zeros(len(y),dtype=np.float32)
    for archive in np.unique(archives):
        for target in [0,1]:
            mask=(archives==archive)&(y==target)
            if not mask.any():
                raise ValueError('a source date lacks one state')
            weights[mask]=1/mask.sum()
    return weights*(len(weights)/weights.sum())


def balanced_brier(y,p):
    if set(np.unique(y)) != {0,1} or np.shape(y) != np.shape(p):
        raise ValueError('balanced Brier requires aligned predictions and both states')
    return float(np.mean([np.mean((p[y==label]-label)**2) for label in [0,1]]))


def augment_source(x,generator):
    """Same shape within a batch; independent uniform sensor subsets per record."""
    b,t,m=x.shape
    if torch.rand((),generator=generator).item()<.5:
        x=x[:,-1000:,:]
    if torch.rand((),generator=generator).item()<.5:
        indices=torch.stack([torch.randperm(m,generator=generator)[:8] for _ in range(b)]).to(x.device)
        x=x.gather(2,indices[:,None,:].expand(b,x.shape[1],8))
    mean=x.mean(1,keepdim=True)
    std=x.std(1,unbiased=False,keepdim=True).clamp_min(1e-12)
    return (x-mean)/std


def predict_binary(model,x,batch=16):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(v).sigmoid() for v in x.split(batch)]).cpu().numpy()


def fit_binary(x,y,archives,spec,training,lr,decay,seed,validation=None,epochs=None,enriched=False):
    if validation is None and epochs is None:
        raise ValueError('final fit needs a source-selected epoch count')
    seed_torch(seed)
    model=make_encoder(spec).to(x.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    target=torch.as_tensor(y,dtype=torch.float32,device=x.device)
    weights=torch.as_tensor(date_state_weights(archives,y),device=x.device)
    maximum=training['maximum_epochs'] if epochs is None else epochs
    best,best_epoch,best_state=float('inf'),0,None
    history=[];start=time.perf_counter()
    for epoch in range(1,maximum+1):
        model.train();losses=[]
        order=torch.randperm(len(x),generator=generator).to(x.device)
        for ix in order.split(training['batch_size']):
            batch=augment_source(x[ix],generator) if enriched else x[ix]
            optimizer.zero_grad(set_to_none=True)
            loss=(nn.functional.binary_cross_entropy_with_logits(model(batch),target[ix],reduction='none')*weights[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite neural training loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        record=dict(epoch=epoch,training_bce=float(np.mean(losses)))
        if validation is not None:
            vx,vy=validation
            score=balanced_brier(vy,predict_binary(model,vx,training['batch_size']))
            record['validation_balanced_brier']=score
            if score<best-1e-8:
                best,best_epoch=score,epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(record)
        if validation is not None and epoch>=training['minimum_epochs'] and epoch-best_epoch>=training['patience']:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch=maximum
    return model,dict(history=history,best_epoch=best_epoch,
        best_validation_brier=None if validation is None else best,epochs_run=len(history),
        seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()))
