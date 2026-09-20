import _pickle as cPickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd

path = 'results/library_df.pkl'
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

alpha = 1/6

shapes = cf.shapes
print(f'=== Original database (with {len(shapes)} datasets)')

obsq = shapes['n_observations'].quantile((alpha,1-alpha)).values
procq = shapes['n_processes'].quantile((alpha,1-alpha)).values

print(f'{100*(1-2*alpha)}% of the data has between {obsq} observations.')
print(f'{100*(1-2*alpha)}% of the data has between {procq} processes.')

sns.set_theme(style="ticks")

f, ax = plt.subplots()
sns.despine(f)
sns.histplot(shapes,x='n_observations',log_scale=True,kde=True,element='step')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([50,100,500,1000])
plt.tight_layout()

f, ax = plt.subplots()
sns.despine(f)
sns.histplot(shapes,x='n_processes',log_scale=True,kde=True,element='step')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([5,10,50])
plt.tight_layout()

plt.show()

lowdim = pd.read_csv('results/lowdim.csv',index_col=0).values
shapes = shapes.loc[[i for i in shapes.index if i not in lowdim]]

min_obs = 100
min_proc = 5
shapes = shapes[(shapes['n_observations'] > min_obs)]
shapes = shapes[(shapes['n_processes'] > min_proc)]
print(f'=== Reduced database (with {len(shapes)} datasets)')

obsq = shapes['n_observations'].quantile((alpha,1-alpha)).values
procq = shapes['n_processes'].quantile((alpha,1-alpha)).values

print(f'{100*(1-2*alpha)}% of the data has between {obsq} observations.')
print(f'{100*(1-2*alpha)}% of the data has between {procq} processes.')

sns.set_theme(style="ticks")

f, ax = plt.subplots()
sns.despine(f)
sns.histplot(shapes,x='n_observations',log_scale=True,kde=True,element='step')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([50,100,500,1000])
plt.tight_layout()

f, ax = plt.subplots()
sns.despine(f)
sns.histplot(shapes,x='n_processes',log_scale=True,kde=True,element='step')
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xticks([5,10,50])
plt.tight_layout()

plt.show()