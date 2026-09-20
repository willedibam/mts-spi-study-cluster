import dill, os

from sklearn.preprocessing import robust_scale
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Helps a bit with speed
dill.settings['byref'] = True
dill.settings['recurse'] = False

basedir = os.path.dirname(os.path.abspath(__file__))

figbase = (5,2)
figbase_dims = (750,10)

for dirpath, _, filenames in os.walk(os.path.join(basedir,'database')):
    for filename in filenames:
        if filename.endswith('.npy'):
            path = os.path.join(dirpath,filename)
            savefilename = os.path.join(dirpath,filename[:-4] + '_scaled')
            # if os.path.exists(savefilename+'.jpg'):
            #     print(f'{savefilename}.jpg and {savefilename}.gif already exist. Skipping.')
            #     continue
            print(f'Generating JPEG for dataset {filename}')

            mvts = filename[:-4]

            try:
                dat = np.load(path)
                if dat.shape[0] > dat.shape[1]:
                    dat = dat.T
                dat = robust_scale(dat)

                
                # Re-scale figure size based on data shape
                figsize = (figbase[0]*dat.shape[1]/figbase_dims[0],figbase[1]*dat.shape[0]/figbase_dims[1])
                fig, ax = plt.subplots(figsize=figsize)

                ax.clear()
                # if 'cauchy' in filename:
                #     vlims = np.percentile(dat,(2,98))
                # else:
                vlims = np.percentile(dat,(5,95))
                vlim = max(vlims)
                vmin, vmax = -vlim, vlim
                ax.pcolormesh(dat,cmap=sns.color_palette('icefire', as_cmap=True),vmin=vmin,vmax=vmax)
                plt.tick_params(which='both', bottom=False,top=False,left=False,right=False,
                                        labelbottom=False,labelleft=False)
                ax.invert_xaxis()
                fig.savefig(savefilename+'.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
                plt.close(fig)
            except ValueError as err:
                print(f'Issue extracting file: {err}')