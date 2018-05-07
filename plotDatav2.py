# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:11:17 2018

@author: monika
"""
#standard modules
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
# custom modules
import seastarV4 as ssv
import style


def loadData(analysisPath, condition):
    """load 3D tracks and backgrounds"""
    dataset = {}
    paramDict =  ssv.readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
    # bg images
    bg1  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS1'))).astype(np.float)
    bg2  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS2'))).astype(np.float)
    ## TODO update for newer 8 coumns datas
    linked3D = plt.loadtxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)))
    try:
        linked3D = np.reshape(linked3D, (-1, paramDict['NStars'], 7))
    except ValueError:
        linked3D = np.reshape(linked3D, (-1, paramDict['NStars'], 8))
    # remove bad points
    qflag = linked3D[:,:,-1]
    indices = np.where(qflag==1)
    linked3D[qflag!=1] = np.ones(8)*np.nan
    tracks2d = linked3D[:,:,:4]
    tracks3d = linked3D[:,:,4:7]
   
     # brightness
    time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition, 'SS1')), unpack=True)
    # highresolution time
    timeHR = np.arange(len(linked3D))/paramDict['framerate']# time in minutes
    
    brightnessHR = np.interp(timeHR, time, brightness)
    entry = [timeHR, brightnessHR, bg1, bg2, tracks2d, tracks3d, paramDict, qflag]
    for kindex,key in enumerate(['frametime', 'brightness', 'bgSS1', 'bgSS2', '2DTracks', '3DTracks', 'pars', 'quality']):
        dataset[key] = entry[kindex]
    return dataset
    
# read all datasets    
paths = ['G:/Data/SeastarData/Analysis/N', 'G:/Data/SeastarData/Analysis/O',\
         'G:/Data/SeastarData/Analysis/P', 'G:/Data/SeastarData/Analysis/T',\
         'G:/Data/SeastarData/Analysis/V']
# always use order single, single, both
conditions = [['Nestle', 'Nutella', 'BothNs'], ['Orzo', 'Okra', 'BothOs'],\
              ['Pomme', 'Persimmon', 'BothPs'], ['Tortellini', 'TiraMisu', 'BothTs'],\
              ['Vermicelli', 'Vanilla', 'BothVs']]
# what to plot
plotLoc = False


Data = {}
for pindex, path in enumerate(paths):
    Data[path] = {}
    for c in conditions[pindex]:
        Data[path][c] = loadData(path, c)

# create a dictionary to store results
results = {}
for pindex, path in enumerate(paths):
    results[path] = {}
    
if plotLoc:    
    colors = ['orange', 'lightblue']      
            
    fig = plt.figure('Location', figsize=(20,9))
    gs = gridspec.GridSpec(len(Data.keys()), 5, width_ratios=[1,1,1,1,0.1])
    # make an illustrative plot with paths and such
    for pindex, path in enumerate(paths):
        for cindex, c in enumerate(conditions[pindex]):
            dset = Data[path][c]
            
            nStars = dset['pars']['NStars']
            
            for n in range(nStars):
                # plot tracks in 2D
                ax = plt.subplot(gs[pindex,cindex+n])
                ax.set_title(c)
                if n==2:
                    ax.set_title('Both Seastar {}'.format(n+1))
                ax.imshow(dset['bgSS1'], cmap='gray')
                cim = ax.scatter(dset['2DTracks'][:,n,0], dset['2DTracks'][:,n,1], s=1, c=dset['3DTracks'][:,n,-1], cmap='inferno_r', vmin=70, vmax=120)
                ax.set_axis_off()
        cb = plt.subplot(gs[pindex,4])
        plt.colorbar(cim, cax=cb)
        cb.set_ylabel('Depth (cm)')
    gs.tight_layout(fig)            
    plt.show()

# calculate velocity and distance
for pindex, path in enumerate(paths):
    veloTmp = []
    coords = []
    distance = []
    brightnessTmp = []
    for cindex, c in enumerate(conditions[pindex]):
        dset = Data[path][c]
        nStars = dset['pars']['NStars']
        # plot tracks in 2D
        for n in range(nStars):
            size = dset['3DTracks'][:,n].T.shape
            
            if size[1]<12900:
                tmpCoor = np.pad(dset['3DTracks'][:,n].T,((0,0),(0,int(12900-size[1]))),mode='constant', constant_values=(np.nan,))
            else:
                tmpCoor = dset['3DTracks'][:,n].T
            # crop tp 12900
            X,Y,Z = tmpCoor[:,:12900]
            
            #print len(X)
            velocity = np.sqrt(np.diff(X)**2 +np.diff(Y)**2+np.diff(Z)**2)*3 # in cm/min
            #velocity = velocity[np.isfinite(velocity)]
            veloTmp.append(velocity)
            coords.append([X,Y,Z])
            b = dset['brightness']
            
            if size[1]<12900:
                b = np.pad(b,(0,12900-size[1]),mode='constant', constant_values=(np.nan,))
            b = b[:12900]
            brightnessTmp.append(b>np.mean(dset['brightness']))
           
    # calculate distance between solo and paired and randomized animals
    
    for i in range(100):
        XS1, YS1, ZS1 = coords[0] #first solo animal
        # shift by oders of minutes:
        #we normally have 12960 frames => shift to cover all = 43 minute shifts
        # roll arrays to get a shift between solo animals 
        shift = 129*i
        tmpC = np.roll(coords[1], shift = shift, axis=1)
        XS2, YS2, ZS2 = tmpC #sec. solo animals
        distS = np.sqrt((XS1 - XS2)**2 +(YS1 - YS2)**2 +(ZS1 - ZS2)**2)
        if i ==0:
            results[path]['distanceS']= [distS]
        else:
            results[path]['distanceS'].append(distS)
        
    XB1, YB1, ZB1 = coords[2] #first of aired animals
    XB2, YB2, ZB2 = coords[3] #second of paired animals
    distB = np.sqrt((XB1 - XB2)**2 +(YB1 - YB2)**2 +(ZB1 - ZB2)**2)
    # randomize order
    distR = []
    for k in range(100):
        tmpCoords = np.array(coords[3]).T
        np.random.shuffle(tmpCoords)
        
        XB2, YB2, ZB2 = tmpCoords.T #second of paired animals
        distR.append(np.sqrt((XB1 - XB2)**2 +(YB1 - YB2)**2 +(ZB1 - ZB2)**2))
    distR = np.nanmean(np.array(distR), axis=0)
    
    results[path]['velocity'] = veloTmp
    
    results[path]['distanceB'] = distB
    results[path]['distanceRnd'] = distR
    results[path]['brightness'] = brightnessTmp
    

fig = plt.figure('Velocities', figsize=(8,8))
gs = gridspec.GridSpec(len(Data.keys()), 2, width_ratios=[1,1])


dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
# make an illustrative plot with paths and such
for pindex, path in enumerate(paths):
    v = results[path]['velocity']
    b = results[path]['brightness']
    # distribution of velocities for each star
    
    #style.mkStyledBoxplot(fig, ax,np.arange(4), v, ['r','b', 'k', 'g'], conditions[pindex])
    
   
    ax = plt.subplot(gs[pindex,0])
    for cindex in range(4):
        hist, bins = np.histogram(v[cindex][np.isfinite(v[cindex])], np.arange(0,10,0.1), normed=True)
        #plt.step()
        #plt.fill_between(bins[:-1]+0.5*np.diff(bins[:2]),y1= hist,lw=2, step='post')#, label=conditions[cindex], color=dcolors[cindex], zorder=-cindex, alpha=0.5)
        plt.plot(bins[:-1]+0.5*np.diff(bins[:2]), hist,lw=2, color=dcolors[cindex], zorder=-cindex, alpha=1, label=plotConditions[cindex])
    plt.xscale('log') 
    plt.xlabel('Velocity (cm/min)')
    plt.legend() 
    dayV = []
    nightV = []
    for ci in range(4):
        dayTmp = v[ci][b[ci][:-1]]
        nightTmp = v[ci][~b[ci][:-1]]
        dayV.append(dayTmp[np.isfinite(dayTmp)])
        nightV.append(nightTmp[np.isfinite(nightTmp)])
        
    
    # make barplot of day and night velocities
    ax = plt.subplot(gs[pindex,1])
    #style.mkStyledBoxplot(fig, ax,np.arange(4), dayV, dcolors, plotConditions, alpha=0.8)
    #style.mkStyledBoxplot(fig, ax,np.arange(4)+0.5, nightV, dcolors, plotConditions, alpha=0.3)
    plt.scatter(np.arange(4), [np.nanmean(dayVel) for dayVel in dayV], color='r', label='day')
    plt.errorbar(np.arange(4), [np.nanmean(dayVel) for dayVel in dayV], [np.nanstd(dayVel) for dayVel in dayV], color='r', linestyle='none')
    plt.scatter(np.arange(4)+0.5, [np.nanmean(nVel) for nVel in nightV], color='C0', label='night')
    plt.errorbar(np.arange(4)+0.5, [np.nanmean(nVel) for nVel in nightV], [np.nanstd(nVel) for nVel in nightV], color='C0', linestyle='none')
    ax.set_ylim([0,10])
    ax.set_xlim([-1,5])
    ax.set_xticks(np.arange(4)+0.25, plotConditions)
    gs.tight_layout(fig)
    
    
    
fig = plt.figure('Distances', figsize=(8,8))
gs = gridspec.GridSpec(len(Data.keys()), 2, width_ratios=[1,1])    
dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Distance (alone)', 'Distance both', 'Distance Random']
# make an illustrative plot with paths and such
for pindex, path in enumerate(paths):
    dS = results[path]['distanceS']
    dB = results[path]['distanceB']
    dR = results[path]['distanceRnd']
    #b = results[path]['brightness']
    bins = np.arange(0,100,1) # distance in cm. 120 is longest axis
    x = bins[:-1]+0.5*np.diff(bins[:2])# xaxis for plots
    # distribution of distances for each star
    # find ranges of solo distance for time shifted (bootstrapped-ish) data
    tmpdistSolo = []
    for distance in dS:
        hist, _ = np.histogram(distance[np.isfinite(distance)], bins, normed=True)
        tmpdistSolo.append(hist)
   
    yerrs = np.nanpercentile(tmpdistSolo, [25,75], axis=0)
    
    # plot the null hypothesis and error bands -- solo distances
    ax = plt.subplot(gs[pindex,0])
    # solo value
    plt.plot(x, np.mean(tmpdistSolo, axis=0),lw=2, color=dcolors[0], zorder=-cindex, alpha=1, label=plotConditions[0])
    #plt.plot(x, yerrs[0],lw=2, color=dcolors[0], zorder=-cindex, alpha=1, label=plotConditions[0])
    # errorbands
    plt.fill_between(x, y2=yerrs[0], y1=yerrs[1], color=dcolors[0], alpha=0.5)
    # distance both in tank
    histB, _ = np.histogram(dB[np.isfinite(dB)], bins, normed=True)
    plt.plot(x, histB,lw=2, color=dcolors[1], zorder=-cindex, alpha=1, label=plotConditions[1])
    # random distance
    # distance both in tank
    histR, _ = np.histogram(dR[np.isfinite(dR)], bins, normed=True)
    plt.plot(x, histR,lw=2, color=style.UCgray[0], zorder=-cindex, alpha=1, label=plotConditions[2])
    #plt.xscale('log') 
    plt.xlabel('Distance (cm)')
    plt.legend() 
    
    ##### Cumulative distributions
    # plot the null hypothesis and error bands -- solo distances
    ax = plt.subplot(gs[pindex,1])
    # solo value
    plt.plot(x, np.cumsum(np.mean(tmpdistSolo, axis=0)),lw=2, color=dcolors[0], zorder=-cindex, alpha=1, label=plotConditions[0])
    # errorbands
    yerrsC = np.nanpercentile(np.cumsum(tmpdistSolo, axis=1), [25,75], axis=0)
    plt.fill_between(x, y2=yerrsC[0], y1=yerrsC[1], color=dcolors[0], alpha=0.5)
    # distance both in tank
    histB, _ = np.histogram(dB[np.isfinite(dB)], bins, normed=True)
    plt.plot(x, np.cumsum(histB),lw=2, color=dcolors[1], zorder=-cindex, alpha=1, label=plotConditions[1])
    # random distance
    # distance both in tank
    histR, _ = np.histogram(dR[np.isfinite(dR)], bins, normed=True)
    plt.plot(x, np.cumsum(histR),lw=2, color=style.UCgray[0], zorder=-cindex, alpha=1, label=plotConditions[2])
    #plt.xscale('log') 
    plt.xlabel('Distance (cm)')
    plt.legend() 
    
    gs.tight_layout(fig)    


fig = plt.figure('Velocities (Time)', figsize=(8,8))
gs = gridspec.GridSpec(len(Data.keys()), 2, width_ratios=[1,1])


dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
# make an illustrative plot with paths and such
for pindex, path in enumerate(paths):
    v = results[path]['velocity']
    b = results[path]['brightness']
    dS = results[path]['distanceS']
    dB = results[path]['distanceB']
    ax = plt.subplot(gs[pindex,0])
    for cindex in range(4):
        plt.plot(v[cindex], color=dcolors[cindex], zorder=-cindex, alpha=1, label=plotConditions[cindex])
#        hist, bins = np.histogram(v[cindex][np.isfinite(v[cindex])], np.arange(0,10,0.1), normed=True)
#        #plt.step()
#        #plt.fill_between(bins[:-1]+0.5*np.diff(bins[:2]),y1= hist,lw=2, step='post')#, label=conditions[cindex], color=dcolors[cindex], zorder=-cindex, alpha=0.5)
#        plt.plot(bins[:-1]+0.5*np.diff(bins[:2]), hist,lw=2, color=dcolors[cindex], zorder=-cindex, alpha=1, label=plotConditions[cindex])
    plt.xlabel('Time (frames)') 
    plt.ylabel('Velocity (cm/min)')
    plt.legend() 
   
    # make line plot o distance over time
    ax = plt.subplot(gs[pindex,1])
    plt.plot(np.nanmean(dS, axis=0), color='C1', zorder=-cindex, alpha=1, label='Distance Alone')
    plt.plot(dB, color='C0', zorder=-cindex, alpha=1, label='Distance Both')
    plt.plot(b[2]*np.max(dB)*1.05)
    plt.fill_between(np.arange(len(dB)), np.zeros(len(dB)), np.ones(len(dB))*np.max(dB)*1.05, where=b[2])
    plt.legend()
    gs.tight_layout(fig)
    
    

fig = plt.figure('Mean distances and velocities', figsize=(8,8))
gs = gridspec.GridSpec(2, 1)


dcolors = [style.UCred[0], style.UCblue[0], style.UCred[1], style.UCblue[1]]
plotConditions = ['Seastar 1 (Alone)', 'Seastar 2 (Alone)', 'Seastar 1 (Both)', 'Seastar 2 (Both)']
# make an illustrative plot with distance box plots
meanVS1,meanVS2, meanVB1,meanVB2, meandS, meandR, meandB = [],[],[], [], [], [], []
for pindex, path in enumerate(paths):
    v = results[path]['velocity']
    dS = results[path]['distanceS']
    dB = results[path]['distanceB']
    dR = results[path]['distanceRnd']
    # solo velocities
    meanVS1.append(np.nanmean(v[0]))
    meanVS2.append(np.nanmean(v[1]))
    # both velocities
    meanVB1.append(np.nanmean(v[2]))
    meanVB2.append(np.nanmean(v[3]))
    meandS.append(np.nanmean(dS))
    meandB.append(np.nanmean(dB))
    meandR.append(np.nanmean(dR))

ax = plt.subplot(gs[0,0])
style.mkStyledBoxplot(fig, ax,np.arange(4), [meanVS1, meanVS2, meanVB1, meanVB2], dcolors, plotConditions, alpha=0.8)
ax = plt.subplot(gs[1,0])
style.mkStyledBoxplot(fig, ax,np.arange(3), [meandS, meandB, meandR], dcolors, ['Solo', 'Both',' Random'], alpha=0.8)
gs.tight_layout(fig)
plt.show()

#    b = results[path]['brightness']
#    ax = plt.subplot(gs[pindex,2])
#    
#    for di, d in enumerate(['distanceS','distanceB','distanceRnd']):
#        dist = np.array(results[path][d])
#        # remove nans
#        dist = dist[np.isfinite(dist)]
#        dist = dist[dist<120]
#        hist, bins = np.histogram(dist, np.arange(10,130,10), normed=True)
#        #plt.step()
#        plt.fill_between(bins[:-1]+0.5*np.diff(bins[:2]),y1= hist,label = d, lw=2, step='mid', color=dcolors[di], zorder=-di, alpha=0.5)
#        plt.xlim([0,120])
#    plt.legend()
#plt.show()

#analysisPath =  '/media/monika/MyPassport/Ns/Analysis'
#data = []
#
#conditions = ['Both', 'Nestle', 'Nutella']
## if both animals ==2, if either one code as 0/1
#code = [2,1,0]
#data = {}
#for cindex, condition in enumerate(conditions):
#    data[].append(loadData(analysisPath, condition))
#
#c = ['#DC143C', '#4876FF']
#
#fig = plt.figure('Location', figsize=(20,9))
#gs = gridspec.GridSpec(3, 4,
#                           width_ratios=[1,1, 2, 1])
#                           
#dataDict = {}
#for dindex, dataSet in enumerate(data):                         
#    
#    linked3D, bg1, bg2, brightness, time, pars = dataSet
#    
#    velocity, distance = calculateProperties(linked3D, pars)
#    
#    
#        
#        
    