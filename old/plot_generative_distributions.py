import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm

plt.rcParams.update({'font.size': 24})

# Illustrate the standard normal distribution
def normal_pdf(mu,sigma,x):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)

def cauchy_pdf(x_0,gamma,x):
    return gamma/(np.pi*(1+((x-x_0)/gamma)**2))

xmin = -3
xmax = 3
N = 48
mu = 0
sigma = 1
x_0 = 0
gamma = 1

figsize = (3,6)

x = np.linspace(xmin,xmax,N)
npdf = normal_pdf(mu,sigma,x)
cpdf = cauchy_pdf(x_0,gamma,x)

norm = Normalize(vmin=-2, vmax=2)
cmap = sns.color_palette('icefire', as_cmap=True)

m = cm.ScalarMappable(norm=norm, cmap=cmap)
c = m.to_rgba(x)

# Create a color if the y axis value is equal or greater than 0
norm = Normalize(vmin=-2, vmax=2)
cmap = sns.color_palette('icefire', as_cmap=True)

fig, ax = plt.subplots(figsize=figsize)
plt.hlines(y=x,xmin=0,xmax=npdf, color=c,linewidth=2,zorder=1)
plt.scatter(npdf,x,color='k',s=15,zorder=2)
plt.xlim((0,0.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('P(x)')
ax.set_ylabel('x')
fig.savefig('plots/normal.pdf',dpi=300,bbox_inches='tight',transparent=True)


fig, ax = plt.subplots(figsize=figsize)
plt.hlines(y=x,xmin=0,xmax=cpdf, color=c,linewidth=2,zorder=1)
plt.scatter(cpdf,x,color='k',s=15,zorder=2)
plt.xlim((0,0.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('P(x)')
ax.set_ylabel('x')
fig.savefig('plots/cauchy.pdf',dpi=300,bbox_inches='tight',transparent=True)