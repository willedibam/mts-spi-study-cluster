"""Pointwise (local) MI: KSG estimator 1 per sample, and Gaussian pointwise MI, for one channel pair."""
import numpy as np
from scipy.special import digamma
from surr_pairs import acf, multivariate_iaaft
K_NN=4
def ksg_local(x,y,w):
    n=len(x); dx=np.abs(x[:,None]-x[None,:]); dy=np.abs(y[:,None]-y[None,:]); d=np.maximum(dx,dy)
    t=np.arange(n); ex=np.abs(t[:,None]-t[None,:])<=w; d=np.where(ex,np.inf,d); eps=np.sort(d,1)[:,K_NN-1]
    nx=np.sum(np.where(ex,False,dx<eps[:,None]),1); ny=np.sum(np.where(ex,False,dy<eps[:,None]),1)
    return digamma(K_NN)+digamma(n)-digamma(nx+1)-digamma(ny+1)
def gauss_local(x,y):
    r=np.corrcoef(x,y)[0,1]; return -0.5*np.log(1-r**2)-(r**2*(x**2+y**2)-2*r*x*y)/(2*(1-r**2)), r
def pair(X,i,j,seed=0):
    X=(X-X.mean(0))/X.std(0); x,y=X[:,i],X[:,j]
    w=min(int(2*np.dot(acf(x),acf(y))),len(x)//10)
    gl,r=gauss_local(x,y); kl=ksg_local(x,y,w)
    S=multivariate_iaaft(X[:,[i,j]],np.random.default_rng(seed),final='spectrum')[0]; S=(S-S.mean(0))/S.std(0)
    gs,rs=gauss_local(S[:,0],S[:,1]); ks=ksg_local(S[:,0],S[:,1],w)
    return dict(x=x,y=y,gl=gl,kl=kl,r=r,gs=gs,ks=ks,rs=rs,sx=S[:,0],sy=S[:,1],w=w)
