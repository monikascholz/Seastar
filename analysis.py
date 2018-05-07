# -*- coding: utf-8 -*-
"""
Created on Mon May 07 12:27:56 2018
Create seastar datasets with identification.
@author: monika
"""
#standard modules
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

# custom modules
import seastarV5 as ssv
import style


def createDataset(analysisPath, conditions):
    """create a dictionary for each fully analyzed dataset. conditions has the three names of the movie dataset eg. ['Nestle', 'Nutella', 'BothNs']."""
    dataset = {}
    Mainkeys = ['Solo1', 'Solo2', 'Both1', 'Both2']
    
    for cindex, condition in enumerate(conditions):
        pars = ssv.readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
        #some rearranging
        linked3D = plt.loadtxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)))
        try:
            linked3D = np.reshape(linked3D, (-1, pars['NStars'], 7))
        except ValueError:
            linked3D = np.reshape(linked3D, (-1, pars['NStars'], 8))
        # pad so all will be at 12960 at the end
        linked3D = np.pad(linked3D,((0,int(12960-linked3D.shape[0])), (0,0), (0,0)),mode='constant', constant_values=(np.nan,))
        try:
            colors = plt.loadtxt(os.path.join(analysisPath, 'colors_{}.txt'.format(condition)))
            colors = np.reshape(colors, (-1, pars['NStars'], 3))
            colors = np.pad(colors,((0,int(12960-colors.shape[0])), (0,0), (0,0)),mode='constant', constant_values=(np.nan,))
        except IOError:
            colors = np.zeros((len(linked3D,pars['NStars'], 3 )))
            pass
        # now we take apart the paired dataset
        for n in range(pars['NStars']):
            dataset[Mainkeys[cindex+n]] = {}
            dataset[Mainkeys[cindex+n]]['name'] = condition
            dataset[Mainkeys[cindex+n]]['pars'] = pars
            dataset[Mainkeys[cindex+n]]['bg1'] = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS1'))).astype(np.float)
            dataset[Mainkeys[cindex+n]]['bg2'] = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS2'))).astype(np.float)
            
            # two dimensional locations in each image
            dataset[Mainkeys[cindex+n]]['x1'] = linked3D[:,n,0]
            dataset[Mainkeys[cindex+n]]['y1'] = linked3D[:,n,1]
            dataset[Mainkeys[cindex+n]]['x2'] = linked3D[:,n,2]
            dataset[Mainkeys[cindex+n]]['y2'] = linked3D[:,n,3]
            # 3D coordinates
            dataset[Mainkeys[cindex+n]]['X'] = linked3D[:,n,4]
            dataset[Mainkeys[cindex+n]]['Y'] = linked3D[:,n,5]
            dataset[Mainkeys[cindex+n]]['Z'] = linked3D[:,n,6]
            # quality flag
            dataset[Mainkeys[cindex+n]]['quality'] = linked3D[:,n,-1]
            # color of each star
            dataset[Mainkeys[cindex+n]]['color'] = colors[:,n,:]
            # calculate a time in reasonable units
            dataset[Mainkeys[cindex]]['time'] =  np.arange(len(linked3D))/dataset[Mainkeys[cindex]]['pars']['framerate']
            # day/night information
            time, _, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition, 'SS1')), unpack=True)
            brightnessHR = np.interp(dataset[Mainkeys[cindex]]['time'], time, brightness)
            dataset[Mainkeys[cindex+n]]['brightness'] = brightnessHR
            dataset[Mainkeys[cindex+n]]['day'] = brightnessHR>np.mean(brightness)
       
            # calculate useful thinsg such as velocity and distance
            X, Y, Z = linked3D[:,n,4:7].T
            dataset[Mainkeys[cindex+n]]['velocity'] =  np.sqrt(np.diff(X)**2 +np.diff(Y)**2+np.diff(Z)**2)*3 # in cm/min
    # correct identification using color information - first calculate centroids in single movies
    meanS1 = np.nanmean(dataset[Mainkeys[0]]['color']/np.linalg.norm(dataset[Mainkeys[0]]['color'], axis=1)[:,None], axis=0)
    meanS2 = np.nanmean(dataset[Mainkeys[1]]['color']/np.linalg.norm(dataset[Mainkeys[1]]['color'], axis=1)[:,None], axis=0)
    # distance between both
    distBoth = np.sqrt((dataset[Mainkeys[2]]['X']-dataset[Mainkeys[3]]['X'])**2+\
            (dataset[Mainkeys[2]]['Y']-dataset[Mainkeys[3]]['Y'])**2+\
            (dataset[Mainkeys[2]]['Z']-dataset[Mainkeys[3]]['Z'])**2)
    # calculate distance of both from solo color centroids
    colDist = []
    for key in Mainkeys[2:]:
        color = dataset[key]['color']
        colorNew = color/np.linalg.norm(color, axis=1)[:,None]
        col = np.abs(np.sum((colorNew -meanS1)**2, axis=1)-np.sum((colorNew -meanS2)**2, axis=1))
        # interpolate nans
        col = np.interp(dataset[Mainkeys[cindex]]['time'], dataset[Mainkeys[cindex]]['time'][np.isfinite(col)], col[np.isfinite(col)])
        colDist.append(gaussian_filter1d(col, 5))
    # find the minimum distance to a the color of Solo1 - Both1 will be the same star as Solo1
    idn = np.nanargmin(np.array(colDist), axis=0)
    plt.subplot(121)
    plt.plot(dataset[Mainkeys[2]]['X'], dataset[Mainkeys[2]]['Y'])
    plt.plot(dataset[Mainkeys[3]]['X'], dataset[Mainkeys[3]]['Y'])
    for coord in ['X', 'Y', 'Z', 'x1', 'y1', 'x2', 'y2']:
        tmp = np.vstack([dataset[Mainkeys[2]][coord],dataset[Mainkeys[3]][coord]])
        dataset[Mainkeys[2]][coord] = tmp[idn]
        dataset[Mainkeys[3]][coord] = tmp[1-idn]
    plt.subplot(122)
    plt.plot(dataset[Mainkeys[2]]['X'], dataset[Mainkeys[2]]['Y'])
    plt.plot(dataset[Mainkeys[3]]['X'], dataset[Mainkeys[3]]['Y'])
    plt.show()
    return dataset
    
# read all datasets    
paths = ['G:/Data/SeastarData/Analysis/N', 'G:/Data/SeastarData/Analysis/O',\
         'G:/Data/SeastarData/Analysis/P', 'G:/Data/SeastarData/Analysis/T',\
         'G:/Data/SeastarData/Analysis/V']
# always use order single, single, both
conditionList = [['Nestle', 'Nutella', 'BothNs'], ['Orzo', 'Okra', 'BothOs'],\
              ['Pomme', 'Persimmon', 'BothPs'], ['Tortellini', 'TiraMisu', 'BothTs'],\
              ['Vermicelli', 'Vanilla', 'BothVs']]


# create a dataset for each pair and both
analysisPath = paths[4]
conditions = conditionList[4]

createDataset(analysisPath, conditions)
    