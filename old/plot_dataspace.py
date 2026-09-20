import os
import _pickle as cPickle
from utils import plotter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import random

random.seed(1)

# reducer = UMAP(random_state=42)
reducer = TSNE(init='pca',learning_rate='auto',random_state=42,perplexity=10)
# reducer = 'umap'

path = os.path.join('results','library_df.pkl')
print(f'Loading CorrelationFrame from {path}...')
with open(path,'rb') as f:
    cf = cPickle.load(f)
print('Done.')

with open('database/database.pkl','rb') as f:
    database = cPickle.load(f)

with open('database/database-archive.pkl','rb') as f:
    database.update(cPickle.load(f))

feature_matrix = cf.get_feature_matrix(sthresh=.95,dthresh=.05)

cfplt = plotter(cf,database=database)
cfplt.dataspace(classes='estimated',reducer=reducer,xlabel='UMAP-1',ylabel='UMAP-2',plot_nas=False,size='processes',include_contour=True)
plt.show()

cfplt = plotter(cf,database=database)
cfplt.dataspace(classes=[
                            ['SelfRegulationSCP1'],['SelfRegulationSCP2'],
                            ['oscillator','fsync'],
                            ['epidemic','cumulative'],
                            ['fmri','mouse'],['fmri','CONTROL'],
                            ['forex'],['stocks'],['FaceDetection']
                            ],
                    reducer=reducer,xlabel='UMAP-1',ylabel='UMAP-2',plot_nas=False,size='processes',include_contour=True)
plt.show()

cfplt = plotter(cf,database=database)
cfplt.dataspace(classes=[
                            ['SelfRegulationSCP1'],['SelfRegulationSCP2'],
                            ['oscillator','fsync'],
                            ['epidemic','cumulative'],
                            ['fmri','mouse'],['fmri','CONTROL'],
                            ['forex'],['stocks'],['FaceDetection'],
                            ['wilson-cowan'],['vanderpol'],['wave-1D'],['wave-2D'],
                            ['defect_turbulence'],['chaotic_brownian_motion_of_defect'],
                            ['hhn'],['kuramoto-sakaguchi']
                            ],
                    reducer=reducer,xlabel='UMAP-1',ylabel='UMAP-2',plot_nas=False,size='processes',include_contour=True)
plt.show()