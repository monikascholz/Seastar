
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:50:38 2017
plot assistant. make pretty plots.
@author: monika
"""
import numpy as np
import matplotlib as mpl
import os
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection


# change axes
axescolor = 'k'
mpl.rcParams["axes.edgecolor"]=axescolor
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
# text
mpl.rcParams["text.color"]='k'
mpl.rcParams["ytick.color"]=axescolor
mpl.rcParams["xtick.color"]=axescolor
mpl.rcParams["axes.labelcolor"]='k'
mpl.rcParams["savefig.format"] ='pdf'
# change legend properties
mpl.rcParams["legend.frameon"]=False
mpl.rcParams["legend.labelspacing"]=0.25
mpl.rcParams["legend.labelspacing"]=0.25
#mpl.rcParams['text.usetex'] =True
mpl.rcParams["axes.labelsize"]=  12
mpl.rcParams["xtick.labelsize"]=  12
mpl.rcParams["ytick.labelsize"]=  12
mpl.rc('font', **{'sans-serif' : 'FiraSans','family' : 'sans-serif'})
mpl.rc('text.latex', preamble='\usepackage{sfmath}')
plt.rcParams['image.cmap'] = 'viridis'

#=============================================================================#
#                           Define UC colors
#=============================================================================#
UCyellow = ['#FFA319','#FFB547','#CC8214']
UCorange = ['#C16622','#D49464','#874718']
UCred    = ['#8F3931','#B1746F','#642822']
UCgreen  = ['#8A9045','#ADB17D','#616530','#58593F','#8A8B79','#3E3E23']
UCblue   = ['#155F83','#5B8FA8','#0F425C']
UCviolet = ['#350E20','#725663']
UCgray   = ['#767676','#D6D6CE']

UCmain   = '#800000'


def mkStyledBoxplot(fig, ax, x_data, y_data, clrs, lbls, alpha = 0.5) : 
    
    dx = min(np.diff(x_data))
   

    for xd, yd, cl in zip(x_data, y_data, clrs) :
        bp = ax.boxplot(yd, positions=[xd], widths = 0.2*dx, \
                        notch=False, patch_artist=True)
        plt.setp(bp['boxes'], edgecolor=cl, facecolor=cl, \
             linewidth=1, alpha=alpha)
        plt.setp(bp['whiskers'], color=cl, linestyle='-', linewidth=1, alpha=1.0)    
        for cap in bp['caps']:
            cap.set(color=cl, linewidth=1)       
        for flier in bp['fliers']:
            flier.set(marker='+', color=cl, alpha=1.0)            
        for median in bp['medians']:
            median.set(color=cl, linewidth=1) 
        jitter = (np.random.random(len(yd)) - 0.5)*dx / 20 
        dotxd = [xd - 0.25*dx]*len(yd) + jitter

        # make alpha stronger
        #ax.plot(dotxd, yd, linestyle='None', marker='o', color=cl, \
        #        markersize=3, alpha=0.5)  
    ymin = min([min(m) for m in y_data])
    ymax = max([max(m) for m in y_data])
    dy = 10 ** np.floor(np.log10(ymin))
    ymin, ymax = ymin-dy, ymax+dy
    xmin, xmax = min(x_data)-0.5*dx, max(x_data)+0.5*dx
    ax.set_xlim(xmin, xmax)        
    ax.set_ylim(ymin, ymax)  
    ax.set_xticks(x_data)

    for loc, spine in ax.spines.items() :
        if loc == 'left' :
            spine.set_position(('outward', 0))  # outward by 5 points
            spine.set_smart_bounds(True)
        elif loc == 'bottom' :
            spine.set_position(('outward', 5))  # outward by 5 points
            spine.set_smart_bounds(True)            
        else :
            spine.set_color('none')  # don't draw spine
    ax.yaxis.set_ticks_position('left') # turn off right ticks
    ax.xaxis.set_ticks_position('bottom') # turn off top ticks
    ax.get_xaxis().set_tick_params(direction='out')
    ax.patch.set_facecolor('white') # ('none')
    ax.set_xticklabels(lbls, rotation=30, fontsize=14)
    
    #ax.set_aspect(2.0 / (0.1*len(lbls)), adjustable=None, anchor=None)
    #ax.set_aspect(0.01 / (len(y_data)), adjustable=None, anchor=None)        
        
            
def plot2DProjections(xS,yS, zS, fig, gsobj, colors = ['r', 'b', 'orange']):
    '''plot 3 projections into 2d for 3dim data sets. Takes an outer gridspec object to place plots.'''
    s, a = 0.05, 0.25
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3,
    subplot_spec=gsobj, hspace=0.25, wspace=0.5)
    ax1 = plt.Subplot(fig, inner_grid[0])
    fig.add_subplot(ax1)
    ax1.scatter(xS,yS,color=colors[0], s=s, alpha = a)
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    
    ax2 = plt.Subplot(fig, inner_grid[1])
    fig.add_subplot(ax2)
    ax2.scatter(xS,zS,color=colors[1], s= s, alpha = a)
    ax2.set_ylabel('Z')
    ax2.set_xlabel('X')
    
    ax3 = plt.Subplot(fig, inner_grid[2])
    fig.add_subplot(ax3)
    ax3.scatter(zS,yS,color=colors[2], s=s, alpha = a)
    ax3.set_ylabel('Y')
    ax3.set_xlabel('Z')
    return [ax1, ax2, ax3]

def multicolor(ax,x,y,z,t,c, threedim = True, etho = False, cg = 1):
    """multicolor plot modified from francesco."""
    lw = 1
    x = x[::cg]
    y = y[::cg]
    z = z[::cg]
    t = t[::cg]
    if threedim:
        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = Line3DCollection(segs, cmap=c, lw=lw)
        if etho:
            lc = Line3DCollection(segs, cmap=c, lw=lw, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection3d(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
        ax.set_zlim(np.min(z),np.max(z))
    else:
        points = np.array([x,y]).transpose().reshape(-1,1,2)
        segs = np.concatenate([points[:-1],points[1:]],axis=1)
        lc = LineCollection(segs, cmap=c, lw=lw)
        if etho:
            lc = LineCollection(segs, cmap=c, lw=lw, norm=ethonorm)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.min(x),np.max(x))
        ax.set_ylim(np.min(y),np.max(y))
    return lc

def circle_scatter(axes, x_array, y_array, radius=0.5, **kwargs):
    """make scatter plot with axis unit radius.(behaves nice when zooming in)"""
    for x, y in zip(x_array, y_array):
        circle = plt.Circle((x,y), radius=radius, **kwargs)
        axes.add_patch(circle)
    return True

def plotHeatmap(T, Y, ax = None):
    """nice looking heatmap for neural dynamics."""
    if ax is None:
        ax = plt.gca()
    cax1 = ax.imshow(Y, aspect='auto', interpolation='none', origin='lower',extent=[0,T[-1],len(Y),0],vmax=2, vmin=-2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Neuron")
    return cax1
    
def plotEigenworms(T, E, label, color = 'k'):
    """make an eigenworm plot"""
    plt.plot(T, E, color = color, lw=1)
    plt.ylabel(label)
    plt.xlabel('Time (s)')

def plotEthogram(ax, T, etho, alpha = 0.5, yValMax=1, yValMin=0, legend=0):
    """make a block graph ethogram for elegans behavior"""
    colDict = {-1:'red',0:'k',1:'green',2:'blue'}
    labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
    #y1 = np.where(etho==key,1,0)
    
    for key in colDict.keys():
        where = np.squeeze((etho==key))
#        if len((etho==key))==0:
#            
#            continue
        plt.fill_between(T, y1=np.ones(len(T))*yValMin, y2=np.ones(len(T))*yValMax, where=where, \
        interpolate=False, color=colDict[key], label=labelDict[key], alpha = alpha)
    plt.xlim([min(T), max(T)])
    plt.ylim([yValMin, yValMax])
    plt.xlabel('Time (s)')
    plt.yticks([])
    if legend:
        plt.legend(ncol=2)
    

