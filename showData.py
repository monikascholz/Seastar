# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:25:09 2017
plot seastar trajectories.
@author: monika
"""
import matplotlib.pylab as plt
import numpy as np
import seastarV4 as ssv
import os
from skimage.transform import EssentialMatrixTransform
from skimage.transform import warp, rotate
from mpl_toolkits.mplot3d import Axes3D


def defineTank(bg1, bg2, paramDict):
    """click on background images to define the aquarium tank inb 3D."""
    nC =4
    plt.figure('Click on the 8 corners of the tank. Front first, starting left bottom, going counter clockwise', figsize=(16,12))
    plt.subplot(111)
    plt.imshow(bg1)
    locs1 = np.array(plt.ginput(nC, timeout=0))
    plt.close()
    
    plt.figure('Click on the corresponding 8 corners of the tank, following the numbers', figsize=(16,12))
    ax = plt.subplot(111)
    plt.imshow(bg2)
    
    ax.scatter(locs1[:,0], locs1[:,1], c='r', marker='x')
    for i in range(len(locs1)):  
        ax.annotate(str(i+1), (locs1[i,0], locs1[i,1]))
    #plt.text()
    locs2 = np.array(plt.ginput(nC, timeout=0))
    plt.close()
    print 'rot', ssv.calculate3DPoint( locs2[:,0], locs2[:,1], locs1[:,0], locs1[:,1], paramDict).T
    paramDict['rotateSS1']=0
    paramDict['rotateSS2']=0
    print 'o, rot', ssv.calculate3DPoint( locs2[:,0], locs2[:,1], locs1[:,0], locs1[:,1], paramDict).T
    return ssv.calculate3DPoint( locs2[:,0], locs2[:,1], locs1[:,0], locs1[:,1], paramDict).T
    
analysisPath =  '/media/monika/MyPassport/Ns/Analysis'
condition = 'Both'
paramDict =  ssv.readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
print paramDict['NStars']
bg1  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS1'))).astype(np.float)
bg2  = ssv.imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,'SS2'))).astype(np.float)

linked3D = plt.loadtxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)))
linked3D = np.reshape(linked3D, (-1, paramDict['NStars'], 7))
print linked3D.shape
#tank = 0
#if tank:
#    tankCoords = defineTank(bg1, bg2, params)
#    np.savetxt(os.path.join(analysisPath, 'Tank_{}.txt'.format(condition)), tankCoords)
#
#tank = np.loadtxt(os.path.join(analysisPath, 'Tank_{}.txt'.format(condition)))
#print tank.shape
#plt.figure()
#plt.subplot(311)
#plt.plot(tank[:,0], tank[:,1])
#plt.xlabel('X')
#plt.ylabel('Y')
#
#plt.subplot(312)
#plt.plot(tank[:,0], tank[:,2])
#plt.xlabel('X')
#plt.ylabel('Z')
#plt.subplot(313)
#plt.plot(tank[:,2], tank[:,1], 'o')
#plt.xlabel('Z')
#plt.ylabel('Y')
#plt.tight_layout()
#plt.show()

trajectory = False
# interpolate 2d trajectories
for n in range(paramDict['NStars']):
    mask = np.isnan(linked3D[:,n,-1])
    linked3D[:,n,4][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,4][~mask])
    linked3D[:,n,5][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,5][~mask])
    linked3D[:,n,6][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked3D[:,n,6][~mask])

if trajectory:
    fig = plt.figure('3d trajectories {}'.format(condition), figsize=(7,12))
    fig.suptitle('{}'.format(condition))
        
    ax = fig.add_subplot(111)
    for n in range(paramDict['NStars']):
        X,Y,Z = linked3D[:,n,4:].T
        ax.plot(X)
        ax.plot(Y)
        ax.plot(Z)
    plt.show()

# plot trajectories
fig = plt.figure('3d trajectories {}'.format(condition), figsize=(7,12))
fig.suptitle('{}'.format(condition))
    
ax = fig.add_subplot(111, projection='3d')
for n in range(paramDict['NStars']):
    X,Y,Z = linked3D[:,n,4:].T
    ax.plot(X,Y,Z)
plt.show()
 # plot residency projections   
fig = plt.figure('3D projections {}'.format(condition), figsize=(7,12))
fig.suptitle('{}'.format(condition))
xmin, xmax, ymin,ymax, zmin, zmax = -45, 45, -30, 30, 60, 120
   
c = ['#001f3f', '#85144b']

for n in range(paramDict['NStars']):
    X,Y,Z = linked3D[:,n,4:].T
    plt.subplot(311)
    plt.plot(X,Y,'o-',color=c[n], alpha=0.01, markersize=10)
    plt.plot(X[0],Y[0],'o-',color='r')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.subplot(312)
    plt.plot(X,Z,'o-',color=c[n], alpha=0.01, markersize=10)
    plt.plot(X[0],Z[0],'o-',color='r')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.xlim([xmin, xmax])
    plt.ylim([zmin, zmax])
    plt.subplot(313)
    plt.plot(Z,Y,'o-',color=c[n], alpha=0.01, markersize=10)
    plt.plot(Z[0],Y[0],'o-',color='r')
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.xlim([zmin, zmax])
    plt.ylim([ymin, ymax])
plt.tight_layout()
plt.show()




fig = plt.figure('2D histograms of residency {}'.format(condition), figsize=(7,12))
fig.suptitle('{}'.format(condition))
xmin, xmax, ymin,ymax, zmin, zmax = -45, 45, -30, 30, 60, 120
   
c = ['#001f3f', '#85144b']

for n in range(paramDict['NStars']):
    X,Y,Z = linked3D[:,n,4:].T
    plt.title('Front')
    plt.subplot(3,paramDict['NStars'],1)
    H, xedges, yedges = np.histogram2d(X,Y, bins=(18,12), normed = True)
    
    im = plt.imshow(H, origin='lower', extent=[min(xedges), max(xedges), min(yedges), max(yedges)], aspect=1, vmin=1/12900.)
    im.cmap.set_under('w')    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.subplot(3,paramDict['NStars'],2)
    plt.title('Bottom')
    H, xedges, yedges = np.histogram2d(X,Z, bins=(18,12), normed = True)
    plt.imshow(H, origin='lower', extent=[min(xedges), max(xedges), min(yedges), max(yedges)],vmin=1/12900.)
    im.cmap.set_under('w')    
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.colorbar()
    plt.subplot(3,paramDict['NStars'],3)
    plt.title('Side')
    H, xedges, yedges = np.histogram2d(Z,Y, bins=(12,12), normed = True)
    plt.imshow(H, origin='lower', extent=[min(xedges), max(xedges), min(yedges), max(yedges)],vmin=1/12900.)
    im.cmap.set_under('w')    
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.colorbar()
plt.tight_layout()
plt.show()


if paramDict['NStars']>1:
    fig = plt.figure('Distance {}'.format(condition), figsize=(7,12))
    fig.suptitle('{}'.format(condition))
    coords0 = linked3D[:,0,4:].T
    coords1 = linked3D[:,1,4:].T
    dist = np.sqrt(np.sum((coords0-coords1)**2,axis=0))
    print dist.shape
    plt.hist(dist, 200)
    plt.show()
    
