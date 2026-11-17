import numpy as np, style, matplotlib.pyplot as plt
from matplotlib.image import imread
PF='#c0392b'
G={3:(518,686),10:(462,644),11:(515,642),12:(551,657),21:(555,627)}
Ge=[(3,12),(3,11),(11,12),(10,11),(12,21)]
C={52:(222,321),53:(286,349),58:(190,264),59:(244,282),85:(505,773),73:(536,820),26:(276,613)}
Ce=[(52,59),(52,58),(52,53)]
fig,axs=plt.subplots(1,2,figsize=(7.2,4.6))
for ax,(f,P,Ed,t) in zip(axs,(('dl/george.jpg',G,Ge,'George'),('dl/chibi.jpg',C,Ce,'Chibi'))):
    ax.imshow(imread(f),cmap='gray'); ax.axis('off'); ax.set_title(f'{t}, propofol hubs',fontsize=9)
    for a,b in Ed: ax.plot(*zip(P[a],P[b]),color=PF,lw=2.2,alpha=.85)
    for e,(x,y) in P.items(): ax.add_patch(plt.Circle((x,y),17,fill=False,ec=PF,lw=1.8))
fig.savefig('fig/hub_brain.png')
