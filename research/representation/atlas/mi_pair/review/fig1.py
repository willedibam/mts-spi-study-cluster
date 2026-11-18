import numpy as np, style, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
COL={'KT':'#d1495b','KTMD':'#edae49','MD':'#00798c','PF':'#30638e','SLP':'#8c8c8c'}
NAME={'KT':'ketamine','KTMD':'ket–med','MD':'medetomidine','PF':'propofol','SLP':'sleep'}
B=np.load('edgemi_nt2s_32.npz'); E=B['E']; unit,state,agent,ids=B['unit'],B['state'],B['agent'],B['id']
mig=lambda r:-0.5*np.log(1-r**2)
def z(a,b):
    a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True); return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
zr=z(mig(E[:,0]),E[:,1]); zs=z(mig(E[:,2]),E[:,3]); R=zr-zs
fig=plt.figure(figsize=(7.4,5.0)); gs=fig.add_gridspec(2,4,height_ratios=[1,1.05],width_ratios=[1,1,1.25,1.25])
# a: edge clouds, one propofol window pair (wake vs drug), real vs IAAFT
u=[x for x in np.unique(unit) if x.startswith('20120802')][0]
def pick(s):
    m=np.where((unit==u)&(state==s))[0]; return m[np.argmin(np.abs(R[m]-np.median(R[m])))]
for i,(s,lab,c) in enumerate((('awake_ec','wake','#555555'),('drug','propofol',COL['PF']))):
    ax=fig.add_subplot(gs[0,i]); k=pick(s)
    ax.scatter(mig(E[k,2]),E[k,3],s=4,c='#cfcfcf',lw=0,label='IAAFT')
    ax.scatter(mig(E[k,0]),E[k,1],s=4,c=c,lw=0,alpha=.8)
    ax.set_title(lab,color=c,fontsize=9); ax.set_xlabel('Gaussian MI'); 
    if i==0: ax.set_ylabel('kNN MI'); ax.text(-.32,1.08,'a',transform=ax.transAxes,fontweight='bold',fontsize=11)
    ax.text(.04,.96,f'z = {zr[k]:.2f}\nIAAFT {zs[k]:.2f}',transform=ax.transAxes,va='top',fontsize=7.5)
    ax.set_xlim(0,1.6); ax.set_ylim(-.15,1.3)
# b: per-session trajectories
ax=fig.add_subplot(gs[0,2:]); xs={'awake_ec':0,'drug':1,'recovery_ec':2,'sleep':1}
for uu in np.unique(unit):
    ag=agent[unit==uu][0]; m=lambda s: R[(unit==uu)&(state==s)].mean()
    st=['awake_ec','sleep'] if ag=='SLP' else [s for s in ('awake_ec','drug','recovery_ec') if ((unit==uu)&(state==s)).any()]
    y=[m(s)-m('awake_ec') for s in st]; ax.plot([xs[s]+(.06 if ag=='SLP' else 0) for s in st],y,'-o',color=COL[ag],ms=2.5,lw=.9,alpha=.85)
eo=[R[(unit==uu)&(state=='awake_eo')].mean()-R[(unit==uu)&(state=='awake_ec')].mean() for uu in np.unique(unit)]
ax.axhspan(-2*np.std(eo),2*np.std(eo),color='#eeeeee',zorder=0); ax.axhline(0,color='#999999',lw=.6)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['wake','drug / sleep','recovery']); ax.set_ylabel('Δz beyond IAAFT')
ax.legend(handles=[Line2D([],[],color=COL[a],lw=1.5,label=NAME[a]) for a in COL],ncol=2,fontsize=7,loc='lower left')
ax.text(-.12,1.08,'b',transform=ax.transAxes,fontweight='bold',fontsize=11)
# c: ladder of form pairs (2-s, beyond IAAFT, in eyes-open SDs)
res=np.load('pairscreen_resid.npy'); labs=['Gaussian MI | kNN MI','r² | ρ²','ρ² | kNN MI','r² | dCor²','|r| | envelope']
ax=fig.add_subplot(gs[1,:2]); DS=[('KT','drug'),('KTMD','drug'),('MD','drug'),('PF','drug'),('SLP','sleep')]
def con(v,ag,s):
    o=[]
    for uu in np.unique(unit[agent==ag]):
        a=(unit==uu)&(state==s); c=(unit==uu)&(state=='awake_ec')
        if a.any() and c.any(): o.append(v[a].mean()-v[c].mean())
    return np.array(o)
for i,v in enumerate(res):
    eo=np.concatenate([con(v,ag,'awake_eo') for ag in COL]); sd=eo.std(ddof=1)
    for j,(ag,s) in enumerate(DS):
        e=con(v,ag,s).mean()/sd; ax.scatter(j,i,s=min(abs(e),4)*45+3,c=COL[ag] if e<-2 else 'white',edgecolors=COL[ag],lw=1)
ax.set_yticks(range(5)); ax.set_yticklabels(labs,fontsize=7.5); ax.invert_yaxis(); ax.set_xticks(range(5)); ax.set_xticklabels([NAME[a] for a,_ in DS],rotation=30,ha='right',fontsize=7.5)
ax.set_xlim(-.6,4.6); ax.spines[['left','bottom']].set_visible(False); ax.tick_params(length=0)
ax.text(-.62,1.04,'c',transform=ax.transAxes,fontweight='bold',fontsize=11)
# d: 10-s replication
D=np.load('decide10_nt.npz'); V=D['V']; N=list(D['names']); u10,s10,a10,df=D['unit'],D['state'],D['agent'],D['delta_fraction']
deep=np.zeros(len(u10),bool)
for uu in np.unique(u10):
    m=(u10==uu)&(s10=='sleep')
    if m.any(): deep[m]=df[m]>=np.quantile(df[m],2/3)
v=V[:,0,N.index('zgk')]-V[:,1,N.index('zgk')]
def con10(sel,ag):
    o=[]
    for uu in np.unique(u10[a10==ag]):
        a=(u10==uu)&sel; c=(u10==uu)&(s10=='awake_ec')
        if a.any() and c.any(): o.append(v[a].mean()-v[c].mean())
    return np.array(o)
eo=np.concatenate([con10(s10=='awake_eo',ag) for ag in COL]); sd=eo.std(ddof=1)
ax=fig.add_subplot(gs[1,2:]); G=[('KT',s10=='drug'),('KTMD',s10=='drug'),('MD',s10=='drug'),('PF',s10=='drug'),('SLP',deep)]
for j,(ag,sel) in enumerate(G):
    c=con10(sel,ag)/sd; ax.scatter(np.full(len(c),j)+np.linspace(-.08,.08,len(c)),c,s=14,color=COL[ag],zorder=3); ax.hlines(c.mean(),j-.25,j+.25,color=COL[ag],lw=2)
ax.axhspan(-2,2,color='#eeeeee',zorder=0); ax.axhline(0,color='#999999',lw=.6)
ax.set_xticks(range(5)); ax.set_xticklabels(['ketamine','ket–med','medetom.','propofol','deep sleep'],rotation=30,ha='right',fontsize=7.5); ax.set_ylabel('Δz beyond IAAFT (null SD)')
ax.set_title('10-s windows',fontsize=8.5,color='#555555')
ax.text(-.12,1.04,'d',transform=ax.transAxes,fontweight='bold',fontsize=11)
fig.savefig('fig/mi_feature.png'); fig.savefig('fig/mi_feature.svg')
