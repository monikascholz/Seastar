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
from scipy import signal, fftpack

# custom modules
import seastarV5 as ssv
import style
plt.ion()

def createDataset(analysisPath, conditions, show=False):
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
            
#        colDist.append(gaussian_filter1d(col, 15))
        except IOError:
            colors = np.zeros((len(linked3D),pars['NStars'], 3 ))
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
            
            # calculate a time in reasonable units
            dataset[Mainkeys[cindex]]['time'] =  np.arange(len(linked3D))/dataset[Mainkeys[cindex]]['pars']['framerate']
            # day/night information
            time, _, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition, 'SS1')), unpack=True)
            brightnessHR = np.interp(dataset[Mainkeys[cindex]]['time'], time, brightness)
            dataset[Mainkeys[cindex+n]]['brightness'] = brightnessHR
            dataset[Mainkeys[cindex+n]]['day'] = brightnessHR>np.mean(brightness)
            # color of each star
            dataset[Mainkeys[cindex+n]]['color'] =  colors[:,n,:]
            dataset[Mainkeys[cindex+n]]['daycolor'] = np.nanmedian(colors[dataset[Mainkeys[cindex+n]]['day'],n,:], axis=0)
            dataset[Mainkeys[cindex+n]]['nightcolor'] = np.nanmedian(colors[~dataset[Mainkeys[cindex+n]]['day'],n,:], axis=0)
            # calculate useful things such as velocity and distance
            X, Y, Z = linked3D[:,n,4:7].T
            dataset[Mainkeys[cindex+n]]['velocity'] =  np.sqrt(np.diff(X)**2 +np.diff(Y)**2+np.diff(Z)**2)*3 # in cm/min
    # correct identification using color information - first calculate centroids in single movies
    meanS1 = np.nanmedian(dataset[Mainkeys[0]]['color'])#/np.linalg.norm(dataset[Mainkeys[0]]['color'], axis=1)[:,None], axis=0)
    meanS2 = np.nanmedian(dataset[Mainkeys[1]]['color'])#/np.linalg.norm(dataset[Mainkeys[1]]['color'], axis=1)[:,None], axis=0)
    # calculate the distance in color space and assign identification
    colDist = []
    for ki, key in enumerate(Mainkeys[2:]):
        color = dataset[key]['daycolor']
        colorNew = color
        colDist.append(np.abs(np.sum((colorNew - meanS1)**2)-np.sum((colorNew -meanS2)**2)))
    
    idn = np.argmin(colDist) # this is the paired animals corresponding to solo1
    # swap all entries
    for coord in dataset[Mainkeys[3]].keys():
        tmp = np.vstack([dataset[Mainkeys[2]][coord],dataset[Mainkeys[3]][coord]])
        dataset[Mainkeys[2]][coord] = tmp[idn]
        dataset[Mainkeys[3]][coord] = tmp[1-idn]
    # assign corresponding names
    dataset[Mainkeys[2]]['name'] =  dataset[Mainkeys[idn]]['name']
    dataset[Mainkeys[3]]['name'] =  dataset[Mainkeys[1-idn]]['name']
    
    ####################################################
    # calculate metrics
    ####################################################
    
    # distance between both
    distBoth = np.sqrt((dataset[Mainkeys[2]]['X']-dataset[Mainkeys[3]]['X'])**2+\
            (dataset[Mainkeys[2]]['Y']-dataset[Mainkeys[3]]['Y'])**2+\
            (dataset[Mainkeys[2]]['Z']-dataset[Mainkeys[3]]['Z'])**2)
    # calculate distance between solo - aligned by day/night, similar to both. roll the second solo
    # find timeshift between Solo1, solo2
    shift = np.argmax(np.correlate(dataset[Mainkeys[0]]['brightness']-np.mean(dataset[Mainkeys[0]]['brightness']),dataset[Mainkeys[1]]['brightness']-np.mean(dataset[Mainkeys[1]]['brightness']),mode='full'))
    distSolo = np.zeros(len(dataset[Mainkeys[0]]['brightness'])) 
    for coord in ['X', 'Y', 'Z']:
        distSolo+=(dataset[Mainkeys[0]][coord]-np.roll(dataset[Mainkeys[1]][coord], shift))**2
    # sqrt for actual distance in cm
    distSolo = np.sqrt(distSolo)
    # bootstrap solo distance
    distSoloBoot = np.zeros((100, len(dataset[Mainkeys[0]]['brightness'])))
    for i in range(100):
        for coord in ['X', 'Y', 'Z']:
            shift = 129*i
            distSoloBoot[i,:] += (dataset[Mainkeys[0]][coord]-np.roll(dataset[Mainkeys[1]][coord], shift))**2
        distSoloBoot[i] = np.sqrt(distSoloBoot[i])
    # randomized distances for both
    distRand = np.zeros((100, len(dataset[Mainkeys[0]]['brightness'])))
    for i in range(100):
        for coord in ['X', 'Y', 'Z']:
            tmp = dataset[Mainkeys[1]][coord]
            np.random.shuffle(tmp)
            distRand[i,:] += (dataset[Mainkeys[0]][coord]-tmp)**2
        distRand[i] = np.sqrt(distRand[i])
        
    dataset['results'] = {}
    dataset['results']['distanceBoth'] = distBoth
    dataset['results']['distanceSolo'] = distSolo
    dataset['results']['distanceSoloBoot'] = distSoloBoot
    dataset['results']['distanceRnd'] = distRand
    # calculate velocities
    for key in Mainkeys:
        dataset['results']['velocity{}'.format(key)] = dataset[key]['velocity']
    
    ####################################################
    # plot stuff
    ####################################################
        
    if show:
        plt.subplot(311)
        plt.plot(dataset[Mainkeys[2]]['X'], label = dataset[Mainkeys[2]]['name'])
        plt.plot(dataset[Mainkeys[3]]['X'], label = dataset[Mainkeys[3]]['name'])
        plt.legend()
        plt.subplot(312)
        plt.plot(dataset[Mainkeys[2]]['X'], label =dataset[Mainkeys[2]]['name'])
        plt.plot(dataset[Mainkeys[3]]['X'], label =dataset[Mainkeys[3]]['name'])
        plt.legend()
        plt.subplot(313)
        plt.plot(distBoth)
        plt.show()
        plt.figure()
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.imshow( dataset[Mainkeys[i]]['bg1'])
            plt.title(dataset[Mainkeys[i]]['name'])
            plt.scatter(dataset[Mainkeys[i]]['x1'][0],dataset[Mainkeys[i]]['y1'][0], s=3, c='r')
        plt.show()
    
   
    return dataset

Mainkeys = ['Solo1', 'Solo2', 'Both1', 'Both2']   
# read all datasets    
paths = ['G:/Data/SeastarData/Analysis/N', 'G:/Data/SeastarData/Analysis/O',\
         'G:/Data/SeastarData/Analysis/P', 'G:/Data/SeastarData/Analysis/T',\
         'G:/Data/SeastarData/Analysis/V']
# always use order single, single, both
conditionList = [['Nestle', 'Nutella', 'BothNs'], ['Orzo', 'Okra', 'BothOs'],\
              ['Pomme', 'Persimmon', 'BothPs'], ['Tortellini', 'TiraMisu', 'BothTs'],\
              ['Vermicelli', 'Vanilla', 'BothVs']]

# for testing
## create a dataset for each pair and both
#analysisPath = paths[0]
#conditions = conditionList[0]
#
#createDataset(analysisPath, conditions)

####################################################
# load data
####################################################
data = {}
for pindex, analysisPath in enumerate(paths):
    data[analysisPath] = createDataset(analysisPath, conditionList[pindex])

####################################################
# rearrange multiple datasets
####################################################
results = {}

for pindex, analysisPath in enumerate(paths):
    dataset = data[analysisPath]['results']
    for key in dataset.keys():
        if pindex==0:   
            results[key] = []
        results[key].append(dataset[key])
        
for key in results.keys():
    results[key] = np.array(results[key])
  

####################################################
# plot velocity distributions/boxplots
####################################################
fig = plt.figure('Velocity')
ax = plt.subplot(121)
dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
style.mkStyledBoxplot(fig, ax,np.arange(4), [np.nanmean(results['velocitySolo1'], axis=1),
                      np.nanmean(results['velocitySolo2'], axis=1), np.nanmean(results['velocityBoth1'], axis=1), np.nanmean(results['velocityBoth2'], axis=1)], dcolors, plotConditions, alpha=0.8)
plt.ylabel('Velocity (cm/min)')

ax = plt.subplot(122)

v1 = np.stack([results['velocitySolo1'], results['velocityBoth1']], axis=2)
v2 = np.stack([results['velocitySolo2'], results['velocityBoth2']], axis=2)
vs = np.concatenate([v1, v2])
vmean = np.nanmean(vs, axis=1)
verr = np.nanstd(vs, axis=1)
xtmp = np.cumsum(np.ones((2,len(vmean))), axis=0)
plt.plot(xtmp, vmean.T, 'o-',color='C2', alpha=0.5 )
plt.xticks([1,2], ['Alone', 'Paired'], rotation =30)
plt.ylabel('Velocity (cm/min)')

plt.show()

####################################################
# plot velocity distributions
####################################################
fig = plt.figure('velocity distribution')
bins = np.arange(0,5,0.05) # distance in cm. 120 is longest axis
x = bins[:-1]+0.5*np.diff(bins[:2])# xaxis for plots
# distribution of distances for each condition

plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
dists = [results['velocitySolo1'], results['velocitySolo2'],results['velocityBoth1'], results['velocityBoth2']]
for i in range(4):
    plt.subplot(2,1,1)#+i)
    dist = dists[i]
    histS = []
    for d in dist:
        hist, _ = np.histogram(d[np.isfinite(d)], bins, normed=True)
        histS.append(hist)
    yerrs = np.nanpercentile(histS, [25,50,75], axis=0)
    plt.plot(x, yerrs[1], dcolors[i], label = plotConditions[i])
    plt.fill_between(x, y2=yerrs[0], y1=yerrs[2], color=dcolors[i], alpha=0.5)
    plt.legend()
    plt.xlabel('Velocity (cm/min)')
    plt.subplot(2,1,2)
    yerrs = np.nanpercentile(np.cumsum(histS, axis=1), [25,50,75], axis=0)
    plt.plot(x, yerrs[1], dcolors[i], label = plotConditions[i])
    plt.fill_between(x, y2=yerrs[0], y1=yerrs[2], color=dcolors[i], alpha=0.5)
    plt.legend()
    plt.xlabel('Velocity (cm/min)')
plt.show()

####################################################
# plot velocity correlations
####################################################
fig = plt.figure('Velocity Correlations')

dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
ax = plt.subplot(121)
plt.xlabel('Seastar 1 (Alone)')
plt.xlabel('Seastar 2 (Alone)')
v1 = results['velocitySolo1']
v2 = results['velocitySolo2']

#plt.errorbar(np.nanmean(v1, axis=1), np.nanmean(v2, axis=1), np.nanstd(v1, axis=1)/np.sqrt(v1.shape[1]), np.nanstd(v2, axis=1)/np.sqrt(v2.shape[1]), color='C2', linestyle='none')
plt.scatter(np.nanmean(v1, axis=1), np.nanmean(v2, axis=1), color='C2', label='Alone')

v1 = results['velocityBoth1']
v2 = results['velocityBoth2']

#plt.errorbar(np.nanmean(v1, axis=1), np.nanmean(v2, axis=1), np.nanstd(v1, axis=1)/np.sqrt(v1.shape[1]), np.nanstd(v2, axis=1)/np.sqrt(v2.shape[1]), color='C3',alpha=0.8, linestyle='none')
plt.scatter(np.nanmean(v1, axis=1), np.nanmean(v2, axis=1), color='C3',alpha=0.8, label ='Paired')
plt.plot([0,7], [0,7], 'k--')
plt.xlabel('Velocity Seastar 1 (cm/min)')
plt.ylabel('Velocity Seastar 2 (cm/min)')
plt.legend()


plt.show()


####################################################
# plot distances distributions/boxplots
####################################################

fig = plt.figure('Distance')
ax = plt.subplot(121)
dcolors = [style.UCorange[0], style.UCgreen[0], style.UCgray[0], style.UCgreen[1]]
plotConditions = ['Solo Distance', 'Paired Distance', 'Random Distance', 'Solo (Bootstrapped)']
style.mkStyledBoxplot(fig, ax,np.arange(4), [np.nanmean(results['distanceSolo'], axis=1), np.nanmean(results['distanceBoth'], axis=1),\
                      np.nanmean(results['distanceRnd'], axis=(1,2)),np.nanmean(results['distanceSoloBoot'], axis=(1,2))], dcolors, plotConditions, alpha=0.8)
plt.ylabel('Distance (cm)')

ax = plt.subplot(122)

v1 = np.stack([results['distanceSolo'],results['distanceBoth']], axis=2)
vmean = np.nanmean(v1, axis=1)
verr = np.nanstd(v1, axis=1)
xtmp = np.cumsum(np.ones((2,len(vmean))), axis= 0)
plt.plot(xtmp, vmean.T, 'o-',color='C3', alpha=0.75)
plt.xticks([1,2], ['Alone', 'Paired'], rotation = 30)
plt.ylabel('Distance (cm)')

plt.show()


####################################################
# plot distance distributions
####################################################
fig = plt.figure('Distance distribution')
bins = np.arange(0,100,2) # distance in cm. 120 is longest axis
x = bins[:-1]+0.5*np.diff(bins[:2])# xaxis for plots
# distribution of distances for each condition

plotConditions = ['Solo Distance', 'Paired Distance', 'Random Distance', 'Solo (Bootstrapped)']
dcolors = [style.UCorange[0], style.UCgreen[0], style.UCgray[0], style.UCgreen[1]]
dists = [results['distanceSolo'], results['distanceBoth'],results['distanceRnd'], results['distanceSoloBoot']]
for i in range(3):
    plt.subplot(2,1,1)#+i)
    dist = dists[i]
    histS = []
    for d in dist:
        hist, _ = np.histogram(d[np.isfinite(d)], bins, normed=True)
        histS.append(hist)
    yerrs = np.nanpercentile(histS, [25,50,75], axis=0)
    plt.plot(x, yerrs[1], dcolors[i], label = plotConditions[i])
    plt.fill_between(x, y2=yerrs[0], y1=yerrs[2], color=dcolors[i], alpha=0.5)
    plt.legend()
    plt.xlabel('Distance (cm)')
    plt.subplot(2,1,2)
    yerrs = np.nanpercentile(np.cumsum(histS, axis=1), [25,50,75], axis=0)
    plt.plot(x, yerrs[1], dcolors[i], label = plotConditions[i])
    plt.fill_between(x, y2=yerrs[0], y1=yerrs[2], color=dcolors[i], alpha=0.5)
    plt.legend()
    plt.xlabel('Distance (cm)')
plt.show()