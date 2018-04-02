# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:24:38 2017
link tracked data to trajectories.
@author: monika
"""

import skimage
import os
from skimage import io, img_as_float
from skimage.color import rgb2gray
import matplotlib.pylab as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

impath= "/media/monika/MyPassport/SS1NutellaSolo/"
analysisPath= "/media/monika/MyPassport/SS1NutellaSolo_Analysis/"



paramDict = {'start':0,
             'end':None,
             'ext' : ".jpg",
            'daylengths':None,
            'framerate':3., #frames per minute
            'starsize':400,
            'minDistance':100 # in pixels per fram10
}

# load coordinates 
tracks = np.load(os.path.join(analysisPath, 'Tracks.npz'))['tracks']
print 'NPoints', len(tracks)
plt.plot(tracks[:,1], tracks[:,2])
plt.show()
linked = {}
index = 0
# keep assigning points to trajectories until all are assigned
while index < np.max(tracks[:,0]):
    print 'step', index#, linked.keys()
    
    # grab all data points from this time points
    dset = tracks[tracks[:,0]==index]
    #print index,  dset
    # jump to next if no points in this tiestep
    if len(dset)==0:
        index+=1
        continue
    # start off by adding all firt points to the linked trajectories as seeds
    if index ==np.min(tracks[:,0]):
        for dindex, data in enumerate(dset):
            linked[dindex] = [data]
        index += 1
        continue
    
    # make a sorted list of keys
    keys = np.sort(linked.keys()) 
    #go through existing linked trajectories if they exist
    dist = np.zeros((len(keys),len(dset)))
    for kindex, key in enumerate(keys):
        for dindex,data in enumerate(dset):
            # last point in the linked trajectory with key
            t0, x0, y0 = linked[key][-1][:3] 
            # point to match to a trajectory
            t1, x1, y1, _,_ = data
            # test if they are temporally close
            if t1-t0>30:
                dist[kindex, dindex] = 30*paramDict['minDistance']
                
            else:
                # calculate the distance from the previous trajectories
                dist[kindex, dindex] = np.sqrt((x0-x1)**2+(y0-y1)**2)/(t1-t0+1)
                
    # use distance matrix to match trajectory and new points
    bestKey, bestValue = np.argmin(dist, axis = 0), np.min(dist, axis = 0) 
    for dindex,data in enumerate(dset):
        # if the best-matching trajectry is close enough, append
        if bestValue[dindex] < paramDict['minDistance']:
            linked[keys[bestKey[dindex]]].append(data)
        # otherwise create a new trajectory
        else:
            linked[np.max(linked.keys())+1] = [data]
    index +=1
    
print linked.keys()
Npoints = 0
for kindex, key in enumerate(linked.keys()):
    
    traj = np.array(linked[key])
    Npoints += len(traj)
    t,x,y,_,_ = traj.T
    plt.subplot(211)
    plt.plot(x,y, 'o')
    plt.subplot(212)
    plt.plot(t,y, 'o')

plt.show()
print 'Npoints', Npoints, len(tracks)
#X = tracks[:,0:3]

#print X.shape
#
#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
#
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms.fit(X)
#labels = ms.labels_
#cluster_centers = ms.cluster_centers_
#
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)
#
#print("number of estimated clusters : %d" % n_clusters_)
#
## select all times for which data exists
#
#        # #############################################################################
## Plot result
#import matplotlib.pyplot as plt
#from itertools import cycle
#
#plt.figure(1)
#plt.clf()
#
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#for k, col in zip(range(n_clusters_), colors):
#    my_members = labels == k
#    cluster_center = cluster_centers[k]
#    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'X', markerfacecolor=col,
#             markeredgecolor='k', markersize=10)
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
    
