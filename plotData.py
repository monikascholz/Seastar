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
# custom modules
import seastarV4 as ssv
import style


def loadData(analysisPath, condition):
    """load 3D tracks and backgrounds"""
   
    paramDict =  ssv.readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
    # bg images
    bg1  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS1'))).astype(np.float)
    bg2  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS2'))).astype(np.float)
   
    linked3D = plt.loadtxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)))
    linked3D = np.reshape(linked3D, (-1, paramDict['NStars'], 7))
    for n in range(paramDict['NStars']):
        mask = np.isnan(linked3D[:,n,-1])
        linked3D[:,n,4][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,4][~mask])
        linked3D[:,n,5][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,5][~mask])
        linked3D[:,n,6][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,6][~mask])
    
     # brightness
    time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition, 'SS1')), unpack=True)
    # highresolution time
    timeHR = np.arange(len(linked3D))/paramDict['framerate']# time in minutes
    
    brightness = np.interp(timeHR, time, brightness)
    print brightness.shape
    return linked3D, bg1, bg2, brightness, timeHR, paramDict
    
def calculateProperties(linked3D, pars):
    """take 3D tracking data and calculate velocity and such"""
    
    nWorms = linked3D.shape[1]
    velocity = np.zeros((nWorms, linked3D.shape[0]))
    
    for n in range(nWorms):
        X,Y,Z = linked3D[:,n,4:].T
        velocity[n][:-1] = np.diff(X)**2+np.diff(Y)**2+np.diff(Z)**2/pars['framerate']
        
    
    distance = np.zeros(linked3D.shape[0])
    if nWorms >1:
        coords0 = linked3D[:,0,4:].T
        coords1 = linked3D[:,1,4:].T
        distance = np.sqrt(np.sum((coords0-coords1)**2,axis=0))
    return velocity, distance

analysisPath =  '/media/monika/MyPassport/Ns/Analysis'
data = []
conditions = ['Both', 'Nestle', 'Nutella']
# if both animals ==2, if either one code as 0/1
code = [2,1,0]
data = {}
for cindex, condition in enumerate(conditions):
    data[].append(loadData(analysisPath, condition))

c = ['#DC143C', '#4876FF']

fig = plt.figure('Location', figsize=(20,9))
gs = gridspec.GridSpec(3, 4,
                           width_ratios=[1,1, 2, 1])
                           
dataDict = {}
for dindex, dataSet in enumerate(data):                         
    
    linked3D, bg1, bg2, brightness, time, pars = dataSet
    
    velocity, distance = calculateProperties(linked3D, pars)
    
    
        
        
    
gs.tight_layout(fig)
plt.show()

fig = plt.figure('Activity as a function of time', figsize=(20,9))
gs = gridspec.GridSpec(3, 4,
                           width_ratios=[1,1, 2, 1])
for dindex, dataSet in enumerate(data):                         
    
    linked3D, bg1, bg2, brightness, time, pars = dataSet
    
    velocity, distance = calculateProperties(linked3D, pars)
    
    for vi, v in enumerate(velocity):
        v = gaussian_filter1d(v, 20)
        ax = plt.subplot(gs[dindex,0])
        ax.set_title(conditions[dindex])
        ax.plot(time, v, alpha=0.9, color=c[vi])
        brightness -= np.min(brightness)
        brightness /= np.max(brightness)
        #ax.plot(brightness*np.max(v))
        
        ax.fill_between(time, 0, 20, where=brightness<np.mean(brightness), color=style.UCgray[1])
        ax.set_ylim([0,20])
        ax.set_ylabel('Velocity (cm/min)')
        ax.set_xlabel('time (min)')
        
        ax = plt.subplot(gs[dindex,1])
        ax.scatter(brightness, velocity[vi], s=1, color =c[vi] , alpha=0.05)
        ax.set_ylim([0,20])
        ax.set_ylabel('Velocity (cm/min)')
        ax.set_xlabel('Brightness (a.u.)')
        
    if len(velocity)>1:
        
        ax = plt.subplot(gs[dindex,2])
        ax.scatter(velocity[0], velocity[1], s=1, color =style.UCmain , alpha=0.05)
        ax.set_ylim([0,20])
        ax.set_ylabel('Velocity star 1 (cm/min)')
        ax.set_xlabel('Velocity star 2 (cm/min)')
        ax.set_ylim([0,20])
        ax.set_xlim([0,20])
        
        
    
gs.tight_layout(fig)
plt.show()