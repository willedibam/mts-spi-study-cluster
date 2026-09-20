# PySPI stuff
try:
    from pynats.calculator import CalculatorFrame, CorrelationFrame
    from pynats.data import Data
except ModuleNotFoundError:
    from pyspi.calculator import CalculatorFrame, CorrelationFrame
    from pyspi.data import Data

# Data wrangling/OS stuff
import os, random
from tokenize import Name
import numpy as np
import pandas as pd
from functools import partial 

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import networkx as nx

# stats/ML stuff
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import robust_scale
import sklearn.cluster as cluster
from scipy.stats import zscore


def plot_stems(y):
    y = zscore(y)
    x = range(y.size)

    # Create a color if the y axis value is equal or greater than 0
    norm = Normalize(vmin=-2, vmax=2)
    cmap = sns.color_palette('icefire', as_cmap=True)

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    c = m.to_rgba(y)
    
    fig, ax = plt.subplots(figsize=(6,2.5))

    # The vertical plot is made using the vline function
    plt.vlines(x=x,ymin=0, ymax=y, color=c,linewidth=2,zorder=1)
    plt.scatter(x,y,color='k',s=15,zorder=2)
    # plt.plot(x,y,color='k',linewidth=1.5,zorder=3)
    
    # Add title and axis names
    plt.axhline(0,color='k',linewidth=2)
    plt.axvline(0,color='k',linewidth=2)
    #utils._despine(ax)
    ax.set_yticks([-2,0,2])

    ylim = 1.1*np.max(np.abs(y))
    ax.set(xlim=(0,y.size),ylim=(-ylim,ylim))
    plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
    plt.axis('off')
    return fig, ax

class plotter(object):

    def __init__(self,cf=None,database=None,savedir=None,flatten_kwargs={},**kwargs):
        if isinstance(cf,CalculatorFrame):
            cf = CorrelationFrame(cf,flatten_kwargs=flatten_kwargs)
        self.cf = cf
        if self.cf is not None:
            self.feature_matrix = self.cf.get_feature_matrix().fillna(0).T

        self.savedir = savedir
        self.kwargs = kwargs

        self.data = {}
        if database is not None:
            for d in database:
                self.data[d] = robust_scale(database[d]['data'].T,axis=1)

    def __call__(self,**kwargs):
        self.kwargs.update(kwargs)
        self.plot()

    def _reweight(x,ts=(0.75,0.5,0.25),ws=(2,0.75,0.1)):
        for t, w in zip(ts,ws):
            if x >= t:
                return w
        return 0

    def _nudge(pos, x_shift, y_shift):
        return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

    def draw_network(self,adj,f=None,squared=False,node_color=None,color_labels=None,labels_on=False,pos=None,layout='spring',seed=1,use_kk=True,savedir=None,ts=None,ws=None,mvts=None,node_kwargs={},edge_kwargs={}):

        if ts is None:
            if adj.shape[0] < 50:
                ts = (0.9,0.7,0.5)
            else:
                vec = adj.values[np.triu_indices(adj.shape[0],1)]
                ts = np.percentile(vec[~np.isnan(vec)],[99,95,90])
        if ws is None:
            ws = (2,0.75,0.1)
        if squared:
            adj = adj**2
            ts = [t**2 for t in ts]

        G = nx.from_pandas_adjacency(adj)
        if f is not None:

            fig, ax = plt.subplots(1,figsize=(10,7))

            if layout == 'spring':
                if pos is None:
                    if use_kk:
                        pos = nx.kamada_kawai_layout(G)
                    else:
                        pos = None
                pos = nx.spring_layout(G,pos=pos,seed=seed,iterations=1000)
            elif pos is None:
                raise ValueError('pos must be included if layout is not spring.')

            weights = [self._reweight(G[u][v]['weight'],ts=ts,ws=ws) for u, v in G.edges()]
            if node_color is not None:
                node_color = [node_color[f] for f in pos]
            nx.draw_networkx_edges(G,pos=pos,ax=ax,edge_color=None,width=weights,**edge_kwargs)

            if mvts is None:
                nx.draw_networkx_nodes(G,pos=pos,ax=ax,node_size=250,
                    edgecolors='k',node_color=node_color,**node_kwargs)

            pos_labels = self._nudge(pos,0,0.08)
            if labels_on:
                nx.draw_networkx_labels(G,pos=pos_labels,ax=ax,font_size=5)
                plt.margins(x=0.4)
            plt.tight_layout()
            title = f'network-{f}'
        else:
            fig, ax = plt.subplots(1,figsize=(10,10))
            weights = [self._reweight(G[u][v]['weight'],ts=ts,ws=ws) for u, v in G.edges()]

            if layout == 'spring':
                if use_kk:
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = None
                pos = nx.spring_layout(G,seed=seed,pos=pos,iterations=1000)
            elif pos is None:
                raise ValueError('pos must be included if layout is not spring.')

            nx.draw_networkx_edges(G,pos=pos,ax=ax,edge_color=None,width=weights,**edge_kwargs)

            if mvts is not None:
                nx.draw_networkx_nodes(G,pos=pos,ax=ax,node_size=150,
                                        edgecolors=[[0.8*c for c in node_color[f]] for f in pos],
                                        linewidths=1,
                                        node_color=[node_color[f] for f in pos],**node_kwargs)


            if labels_on:
                _ = nx.draw_networkx_labels(G,pos=pos,ax=ax,font_size=1)
                plt.margins(x=0.4)
            title = 'network'

        ax = plt.gca()
        if mvts is not None:
            width = 0.2 # Length of MVTS image
            height = 0.13 # Height of MVTS image
            for p in pos:
                try:
                    Z = mvts[p]['data'].T
                    xmin = pos[p][0]
                    ymin = pos[p][1]
                    Y, X = np.mgrid[ymin:ymin+height:height/Z.shape[0],xmin:xmin+width:width/Z.shape[1]]
                    X = X - width / 2
                    Y = Y - height / 2
                    ax.pcolormesh(X,Y,Z,cmap=sns.color_palette('icefire_r', as_cmap=True))
                except KeyError:
                    print(f'Cannot find data matrix for {p}')
            
        if color_labels is not None:
            ns = []
            for l in color_labels:
                ns.append(ax.scatter([],[],color=color_labels[l],label=l))

            lines = []
            for t, w in zip(ts,ws):
                if squared:
                    l, = ax.plot([],[],color='k',linewidth=w,label=f'|r| > {np.sqrt(t):.2f}')
                else:
                    l, = ax.plot([],[],color='k',linewidth=w,label=f'|r| > {t:.2f}')
                lines.append(l)
            legend1 = plt.legend(lines,[l.get_label() for l in lines],loc=3)
            ax.legend(ns,[n.get_label() for n in ns],loc=1)
            ax.add_artist(legend1)

        # plt.tight_layout()
        plt.axis('off')
        if savedir is not None:
            path = os.path.join(savedir,title+'.pdf')
            fig.savefig(path,dpi=300)
            path = os.path.join(savedir,title+'.png')
            fig.savefig(path,dpi=300)
            print(f'Saving network to {path}.')
            plt.close(fig)

    def plot_clusters(mm_adj,cols=None,col_labels=None,method='average',min_rho=None,apx='',mask_on=False,savedir=None):
        mask = mm_adj.isnull()
        mm_adj[mask] = 0

        if min_rho is not None:
            vmin = min_rho
        else:
            vmin = np.min(mm_adj.values)

        y = 1-mm_adj.abs().values[np.triu_indices(mm_adj.shape[0],1)]
        Z = linkage(y,metric='euclidean',method=method,optimal_ordering=True)

        fig, ax = plt.subplots(figsize=(3, 16))
        dn = dendrogram(Z,labels=mm_adj.columns.values,orientation='left',
                            color_threshold=0,count_sort='ascending',above_threshold_color='k')
        plt.axis('off')

        if savedir is not None:
            fig.savefig(savedir+'/dendrogram-sm' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
            plt.close(fig)

        # The average (spearman) correlation between measures
        mm_adj[mask] = np.NaN
        mm_adj_sort = mm_adj.loc[dn['ivl'],dn['ivl']]

        if mask_on:
            mm_adj_sort.values[np.triu_indices(mm_adj_sort.shape[0], 1)] = np.nan

        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.pcolormesh(mm_adj_sort.values,vmin=vmin,vmax=1,
                            cmap=plt.cm.get_cmap('RdYlBu_r',9))
        plt.tick_params(which='both', bottom=False,top=False,left=False,right=False,
                        labelbottom=False,labelleft=False)
        ax.invert_yaxis()
        plt.axis('off')
        if savedir is not None:
            fig.savefig(savedir+'/mm_cluster' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(1,8))
        fig.colorbar(im,cax=ax)
        ax.tick_params(labelsize=30)
        if savedir is not None:
            fig.savefig(savedir+'/colorbar' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
            plt.close(fig)

        if cols is not None:
            row_cols = np.array([cols[f] for f in dn['ivl']])
            fig, ax = plt.subplots(figsize=(12,2))
            im = ax.imshow(np.repeat(row_cols.reshape((1,len(cols),3)),15,axis=0))
            plt.axis('off')
            if savedir is not None:
                fig.savefig(savedir+'/colorrow' + apx + '.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
                plt.close(fig)

    def rasterplot(data,cmap='icefire',window=7,proc_cluster=True,animate=True,savefilename=None):

        if isinstance(data,np.ndarray):
            data = Data(data)
        dat = data.to_numpy(squeeze=True)

        if animate:
            figsize=(10,10)
            dendrogram_ratio = 0.1
            cbar_pos=None
        else:
            figsize=(7,10)
            cbar_pos=(0, .2, .03, .4)
            dendrogram_ratio = 0.1

        g = sns.clustermap(np.transpose(dat),
                                cmap=cmap,figsize=figsize,
                                col_cluster=proc_cluster,row_cluster=False,
                                dendrogram_ratio=dendrogram_ratio,cbar_pos=cbar_pos,
                                robust=True)
        ax_im = g.ax_heatmap
        fig = ax_im.figure

        ax_im.set_xlabel('Process')
        ax_im.set_ylabel('Time')
        ax_im.figure.suptitle(f'Space-time amplitude plot for "{data.name}"')

        if animate:
            g.gs.update(left=0.05, right=0.45, bottom=0.1, top=0.9)
            gs2 = gridspec.GridSpec(1,1, left=0.6, right=0.9, bottom=0.3, top=0.55)
            ax_st = g.fig.add_subplot(gs2[0,0])

            cols = sns.color_palette('Blues',n_colors=window)
            lines = []
            for t in range(window):
                lines.append(ax_st.plot(dat[:,t],color=cols[t])[0])

            def update_plots(ti,data,lines,ax):
                maxT = data.shape[1]
                for t, line in enumerate(lines):
                    line.set_ydata(data[:,(ti+t)%maxT])
                ax.set_title(f'Amplitude at time t={ti}')

            lims = [np.min(dat),np.max(dat)]
            padding = np.ptp(lims)*0.05
            ax_st.set_ylim([lims[0]-padding,lims[1]+padding])
            ax_st.set_title('Time t=0')
            ax_st.set_xlabel('Process')
            ax_st.set_ylabel('Amplitude')

            repeat = True
            if savefilename is not None:
                repeat = False
            line_ani = animation.FuncAnimation(fig,update_plots,data.n_observations,
                                                fargs=(dat,lines,ax_st),interval=100,blit=False,repeat=repeat)

            ax_im.locator_params(axis='y', nbins=6)
            if savefilename is not None:
                fname = savefilename+'.gif'
                line_ani.save(fname, writer='imagemagick', fps=10)
                print(f'Saved gif to {fname}')
                plt.close(fig)
            else:
                plt.show()
        else:
            if savefilename is not None:
                fname = savefilename+'.jpg'
                fig.savefig(fname,format='jpg',bbox_inches='tight')
                print(f'Saved figure to {fname}')
                plt.close(fig)
            else:
                plt.show()

    def mm_cluster(self,dropna=True,absolute=False,classes=None,flatten_kwargs={},
                    clustermap_kwargs={'cmap': plt.cm.get_cmap('RdYlBu_r',9), 'xticklabels': 1,'yticklabels': 1}):

        mdf = self.cf.mdf

        if absolute:
            mdf = mdf.abs()

        mm_adj = mdf.groupby(level='Source statistic').mean()
        mm_adj = mm_adj.sort_index().reindex(sorted(mm_adj),axis=1)
        
        if dropna:
            mm_adj = mm_adj.dropna(how='all',axis=0).dropna(how='all',axis=1)
            
        if absolute:
            mm_adj = mm_adj.abs()
            clustermap_kwargs['vmin'] = 0
            clustermap_kwargs['vmax'] = 1
        else:
            clustermap_kwargs['vmin'] = -1
            clustermap_kwargs['vmax'] = 1
        mm_adj.fillna(0,inplace=True)

        colors = None
        if classes is not None:
            self.cf.set_sgroups(classes)
            groups = pd.Series(self.cf.get_sgroup_names(mm_adj.columns))
            lut = dict(zip(groups.unique(),sns.color_palette('pastel', groups.unique().size)))
            colors = groups.map(lut).values

        if mm_adj.shape[0] > 20:
            sns.set(font_scale=0.5)
        g = sns.clustermap(mm_adj,col_colors=colors,row_colors=colors,**clustermap_kwargs)

        # Prettify
        ax = g.ax_heatmap
        ax_hmcb = g.ax_cbar
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        sns.set(font_scale=1)
        g.gs.update(top=0.9)
        g.fig.suptitle(f'Measure-measure clustermap for all {self.cf.ddf.shape[1]} datasets of "{self.cf.name}" frame')
        ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])
        
        return ax.figure

    def dd_cluster(self,absolute=True,clustermap_kwargs={'cmap':plt.cm.get_cmap('RdYlBu_r',9),'xticklabels': 1,'yticklabels' :1},
                    classes=None):

        # self.feature_matrix = self.cf.get_self.feature_matrix()
        dd_adj = self.feature_matrix.corr(method='spearman')
            
        mask = dd_adj.isna()
        dd_adj.fillna(0,inplace=True)
        if absolute:
            dd_adj = dd_adj.abs()
            clustermap_kwargs['vmin'] = 0
            clustermap_kwargs['vmax'] = 1
        else:
            clustermap_kwargs['vmin'] = -1
            clustermap_kwargs['vmax'] = 1

        colors = None
        if classes is not None:
            self.cf.set_dgroups(classes)
            groups = pd.Series(self.cf.get_dgroup_names(dd_adj.columns))
            lut = dict(zip(groups.unique(),sns.color_palette('pastel', groups.unique().size)))
            colors = groups.map(lut).values

        if dd_adj.shape[0] > 20:
            sns.set(font_scale=0.5)
        g = sns.clustermap(dd_adj,mask=mask,row_colors=colors,col_colors=colors,**clustermap_kwargs)

        # Prettify
        ax = g.ax_heatmap
        ax_hmcb = g.ax_cbar
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        sns.set(font_scale=1)
        g.gs.update(top=0.9)
        g.fig.suptitle(f'Data-data Clustermap for all {self.cf.ddf.shape[1]} datasets of "{self.cf.name}" frame')
        ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])
        
        return ax.figure

    def _get_reducer(self,reducer):
        random_state = 42
        if reducer == 'pca':
            reducer = PCA(n_components=2, svd_solver='full',random_state=random_state)
            xlabel, ylabel = ('PC-1','PC-2')
        elif reducer == 'umap':
            from umap import UMAP
            reducer = UMAP(random_state=random_state)
            xlabel, ylabel = ('UMAP-1','UMAP-2')
        elif reducer == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2,random_state=random_state)
            xlabel, ylabel = ('tSNE-1','tSNE-2')
        self.reducer, self.xlabel, self.ylabel = reducer, xlabel, ylabel

    # Should really combine this and dataspace
    def measurespace(self,reducer='pca',classes=None,absolute=True):
        mdf = self.cf.mdf

        if absolute:
            mdf = mdf.abs()

        X = mdf.groupby('Source statistic').mean().fillna(0)
        X = X.sort_index().reindex(sorted(X),axis=1)
        
        if reducer != 'eig' and reducer != 'mds':
            reducer, xlabel, ylabel = self._get_reducer(reducer)
            try:
                if absolute:
                    embedding = reducer.fit_transform(np.abs(X))
                else:
                    embedding = reducer.fit_transform(X)
            
                if isinstance(reducer,PCA):
                    xlabel += f' ({100*reducer.explained_variance_ratio_[0]:.2f}%)'
                    ylabel += f' ({100*reducer.explained_variance_ratio_[1]:.2f}%)'
            except ValueError as err:
                print(f'Dimensionality reduction failed: {err}.')
        elif reducer == 'mds':
            embedding = MDS(dissimilarity='precomputed').fit_transform(1-np.abs(X))
            # variance_explained = 1 - np.corrcoef(pdist(X),pdist(embedding))**2
            xlabel = 'MDS-1'
            ylabel = 'MDS-2'
        elif reducer == 'eig':
            B = zscore(X)
            B = np.nan_to_num(B)
            C = np.corrcoef(B)
            v, V = np.linalg.eig(C)
            order = np.argsort(v)[::-1]
            v = v[order]
            V = V[order]
            T = np.matmul(B,V)
            embedding = T[:,:2]
            xlabel = f'EV-1'
            ylabel = f'EV-2'
        else:
            xlabel = 'LD-1'
            ylabel = 'LD-2'

        self.embeddf = pd.DataFrame({xlabel: embedding[:,0], ylabel: embedding[:,1], 'measure': mdf.columns.tolist()},index=mdf.columns)
        
        if classes is not None:
            self.cf.set_sgroups(classes)
            self.embeddf['class'] = self.cf.get_sgroup_names(mdf.columns)

        self.fig, self.ax = plt.subplots(1,1)
        try:
            sns.scatterplot(data=self.embeddf,x=xlabel,y=ylabel,hue='class',palette='pastel')
        except ValueError as err:
            sns.set(font_scale=0.5)
            sns.scatterplot(data=self.embeddf,x=xlabel,y=ylabel,hue='measure',style='measure',palette='pastel')
            sns.set(font_scale=1.0)

    def dataspace(self,classes=None,reducer='pca',absolute=True,size=None,plot_nas=True,feature_set=None,
                        xlabel=None,ylabel=None,include_contour=False,cmap=None,scatterplot_kwargs={}):
        

        if feature_set is not None:
            self.feature_matrix = self.feature_matrix[feature_set]
        
        if isinstance(reducer,str):
            self._get_reducer(reducer)
        else:
            if xlabel is None or ylabel is None:
                raise ValueError('If using a bespoke reducer, include the xlabel and ylabel arguments.')

        try:
            if absolute:
                embedding = reducer.fit_transform(self.feature_matrix)
            else:
                embedding = reducer.fit_transform(self.feature_matrix.abs())

            if isinstance(reducer,PCA):
                self.xlabel += f' ({100*reducer.explained_variance_ratio_[0]:.2f}%)'
                self.ylabel += f' ({100*reducer.explained_variance_ratio_[1]:.2f}%)'

            self.embeddf = pd.DataFrame(index=self.feature_matrix.index,data=embedding,columns=[xlabel,ylabel])
        except ValueError as err:
            raise ValueError(f'Dimensionality reduction failed: {err}.')

        if classes is not None:
            if classes == 'n_processes':
                self.embeddf['class'] = self.cf.shapes.loc[self.embeddf.index]['n_processes']
            elif classes == 'n_observations':
                self.embeddf['class'] = self.cf.shapes.loc[self.embeddf.index]['n_observations']
            elif classes == 'estimated':
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=5)
                self.embeddf['class'] = [f'cluster {c}' for c in clusterer.fit_predict(embedding)]
                self.embeddf[self.embeddf['class'] == 'cluster -1'] = 'N/A'
            else:
                self.cf.set_dgroups(classes)
                self.embeddf['class'] = self.cf.get_dgroup_names(self.feature_matrix.index)

        if not plot_nas:
            self.embeddf = self.embeddf[self.embeddf['class'] != 'N/A']

        def _replot(self,event,scatterfn=None,contourfn=None):

            try:
                sax = self.axs[0]
                dax = self.axs[1]
            except TypeError:
                sax = self.axs
            
            sax.clear()
            calpha = 0.3

            # Re-sort
            self.embeddf['alpha'] = 1.

            if event is not None:
                calpha = 0.1

                # Select only the closest point
                ind = event.ind
                if len(ind) > 1:
                    data_ind = np.array([self.embeddf.iloc[i,:2].values for i in ind],dtype=float)
                    ms = np.array([event.mouseevent.xdata, event.mouseevent.ydata],dtype=float)
                    dist = np.sqrt((data_ind[:,0]-ms[0])**2+(data_ind[:,1]-ms[1])**2)
                    ind = ind[np.argmin(dist)]
                else:
                    ind = ind[0]

                dataset = self.embeddf.index[ind]
                print(f'Clicked datapoint {dataset}')

                # Plot the raster for the clicked dataset
                try:
                    dax.clear()
                    Z = self.data[dataset]
                    vmin, vmax = np.percentile(Z,(1,99))
                    dax.pcolormesh(Z,cmap=sns.color_palette('icefire', as_cmap=True),vmin=vmin,vmax=vmax)
                    dax.set_title(dataset,fontsize=8)
                    dax.set_xlabel('Time',fontsize=7)
                    dax.set_ylabel('Process',fontsize=7)
                except KeyError:
                    pass

                # Make all datapoints transparent except the clicked one
                self.embeddf['alpha'] = [.15] * self.embeddf.shape[0]
                self.embeddf.loc[self.embeddf.index == dataset,'alpha'] = 1.

                # Re-sort the dataframe so that the selected point is at the top
                self.embeddf = self.embeddf.sort_values(by=['alpha'],ascending=True)

            if contourfn is not None:
                try:
                    ax = contourfn(ax=sax,data=self.embeddf,hue='class',hue_order=self.hue_order)
                except ValueError:
                    ax = contourfn(ax=sax,data=self.embeddf,hue=None)
                for c in ax.collections[:]:
                    c.set_alpha(calpha)

            try:
                scatterfn(ax=sax,data=self.embeddf,alpha=self.embeddf['alpha'].values.tolist(),hue='class',hue_order=self.hue_order)
            except ValueError:
                scatterfn(ax=sax,data=self.embeddf,alpha=self.embeddf['alpha'].values.tolist(),hue=None)

            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            sns.move_legend(
                sax, "lower center",
                bbox_to_anchor=(.5, 1), ncol=2, title=None,
            )

            sax.xaxis.set_picker(True)
            sax.xaxis.set_pickradius(.01)
            sax.yaxis.set_picker(True)
            sax.yaxis.set_pickradius(.01)
            try:
                dax.figure.canvas.draw_idle()
            except UnboundLocalError:
                pass

        if len(self.data) > 0:
            self.fig, self.axs = plt.subplots(ncols=1,nrows=2,figsize=(7,10),gridspec_kw={'height_ratios': [3, 1]})
            scatterplot_kwargs.update(dict(picker=4))
        else:
            self.fig, self.axs = plt.subplots(ncols=1,nrows=1,figsize=(7,7))

        self.hue_order = self.embeddf['class'].unique()

        self.cmap = cmap
        if self.cmap is None and len(self.hue_order) > 10:
                cmap = sns.color_palette("hls", len(self.hue_order))
                # Randomize the order of the colors to make it easier to see
                cmap = cmap.as_hex()
                random.seed(3)
                random.shuffle(cmap)
                self.cmap = sns.color_palette(cmap)

        if size is not None:
            if size == 'observations':
                self.embeddf[size] = self.cf.shapes.loc[self.embeddf.index]['n_observations']
                
            elif size == 'processes':
                self.embeddf[size] = self.cf.shapes.loc[self.embeddf.index]['n_processes']

            scatterfn = partial(sns.scatterplot,x=xlabel,y=ylabel,alpha=0.8,size=size,sizes=(10,50),**scatterplot_kwargs,palette=self.cmap)
        else:
            scatterfn = partial(sns.scatterplot,x=xlabel,y=ylabel,alpha=0.8,**scatterplot_kwargs,palette=self.cmap)

        contourfn = None
        if include_contour:
            # Intended to plot the decision boundaries but it's quite painful with multiple classes
            contourfn = partial(sns.kdeplot,x=xlabel,y=ylabel,alpha=0.5,shade=True,bw_adjust=1.3,levels=5,palette=self.cmap)

        plot_scatter = partial(_replot,self,scatterfn=scatterfn,contourfn=contourfn)
        plot_scatter(None)

        try:
            self.axs[1].annotate(f'Click a dot to display the corresponding dataset', (0.5, 0.5),
                                    transform=self.axs[1].transAxes,
                                    ha='center', va='center', fontsize=14,
                                    color='darkgrey')
            self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.fig.canvas.mpl_connect('pick_event', plot_scatter)
        except TypeError:
            pass

    def compute_intrinsic_dimensionality(X,alpha=0.95):
        pca = PCA(n_components=X.shape[0])
        pca.fit(X)
        return np.argmax(pca.explained_variance_ratio_.cumsum() > alpha)+1
        
    def relate(self, stat0, stat1, absolute=False, classes=None, include_total=True):

        if classes is not None:
            self.cf.set_dgroups(classes)

        s0mdf = self.cf.mdf[stat0].reset_index()
        smdf = s0mdf[s0mdf['Source statistic'] == stat1]

        name = f'r({stat0},{stat1})'
        smdf = smdf.rename(columns={stat0: name})

        if absolute:
            smdf[name] = smdf[name].abs()

        _, ax = plt.subplots()
        if classes is not None:
            smdf['class'] = self.cf.get_dgroup_names(smdf['Dataset'])
            smdf_l = smdf[smdf['class'] != 'N/A']
            sns.histplot(smdf_l,x=name,hue='class',stat='probability',common_norm=True, multiple='stack')
            if include_total:
                sns.histplot(smdf,x=name,stat='probability',element="step",alpha=.2)
                ax.axvline(smdf[name].mean(),c='k',ls='--')
            ax.axvline(smdf_l[name].mean(),c='k',ls='-')
        else:
            sns.histplot(smdf,x=name,stat='probability',element="step",common_norm=True, multiple='stack')
            ax.axvline(smdf[name].mean(),c='k',ls='--')
        return smdf

    # For now just take in the dataframe computed by cf.correlation_matrix()
    def concensusmap(self,df,n_clusters=8):
        cdf = pd.DataFrame(columns=df.columns)
        for _, new_df in df.groupby(level=0):
            kmeans = cluster.KMeans(n_clusters).fit(new_df.values)
            ndf = pd.DataFrame(data=kmeans.labels_,index=new_df.name,columns=cdf.columns)
            cdf = cdf.append(ndf)