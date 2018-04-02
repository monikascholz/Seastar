# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:57:34 2017
seastar tracking with separate linkage and separable steps,
@author: monika
"""

import skimage
import datetime
from skimage import io, img_as_float
from skimage.color import rgb2gray
import matplotlib.pylab as plt
import numpy as np
from skimage.filters import gaussian, try_all_threshold
from skimage.feature import blob_doh
import os, re
from skimage.filters import threshold_yen, threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, medfilt
from skimage.transform import warp, rotate
from skimage.filters.rank import median
from skimage.morphology import disk
#=============================================================================
#
#       I/O
#
#=============================================================================

def natural_sort(liste):
    """Natural sort to have frames in right order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(liste, key = alphanum_key)


def loadSubset(impath, extension, start = 0, end = None, step = 1):
    """load images with file extension extension following the pattern given by tuple.
    example: loadSubset('.', 'png', start=0, end=100, step =4) loads every 4th image in the directory up to the hundredth image."""
    imFileList = natural_sort([os.path.join(impath,f) for f in os.listdir(impath) if f.endswith(extension)])[start:end:step]
    return np.array(imFileList)

def imread_convert(f, flag='SS1', rgb=False):
    if rgb:
        if flag=='SS2':
            return img_as_float(io.imread(f))[::-1,::-1,0]
        return img_as_float(io.imread(f))[:,:,0]
    else:
        if flag=='SS2':
            return img_as_float(io.imread(f, as_grey=True))[::-1,::-1]
        return img_as_float(io.imread(f, as_grey=True))

def writeStatus(fname, action):
    """write current analysis steps in a status file. action is a string"""
    with open(fname, 'a') as f:
        now = datetime.datetime.now()
        f.write("{} {}\n".format(now.strftime("%Y-%m-%d %H:%M"),action))

def writePars(fname, parDict):
    """write the parameter dictionary to file."""
    with open(fname, 'w') as f:
        for key in parDict.keys():
            f.write("{} {}\n".format(key, parDict[key]))
            
def readPars(fname):
    """write the parameter dictionary to file."""
    parDict = {}
    with open(fname, 'r') as f:
        for line in f:
            key, value = line.split()
            try:
                parDict[key] = np.float(value)
                if parDict[key].is_integer():
                     parDict[key] = int(parDict[key])
            except ValueError:
                if value=="True" or value=="False":
                    parDict[key] = bool(value)
                else:
                    parDict[key] = value
    return parDict

def loadTracks(analysisPath, condition):
    tracks= {}
    for flag in ['SS1', 'SS2']:
            with open(os.path.join(analysisPath, 'Tracks_{}_{}.txt'.format(condition, flag)), 'r') as f:
                # drop header
                f.next()
                for line in f:
                    if flag =='SS1':
                        tracks[int(line.split()[0])] = []
                    tracks[int(line.split()[0])].append([float(x) for x in line.split()[1:]])
    return tracks
    
#=============================================================================
#
#      3D calculations
#
#=============================================================================  
def rotatePoints(x,y,paramDict, flag):
    """calculate points after image rotation."""
    theta = paramDict['rotate{}'.format(flag)]
    c, s = np.cos(theta), np.sin(theta)
    try:
        T = np.tile(np.array([paramDict['imWidth']/2., paramDict['imHeight']/2.]), (len(x),1)).T
    except TypeError:
        # deal with running floats
        T = np.tile(np.array([paramDict['imWidth']/2., paramDict['imHeight']/2.]), (1,1)).T
    R = np.matrix(np.array([[c,s], [-s, c]]))
    #print np.vstack([x,y]).shape, T.shape
    xn, yn = np.array(np.dot(R, np.vstack([x,y])-T)+T)
    
    return xn, yn
      
def calculate3DPoint(x1, y1, x2, y2, paramDict):
    """single point parallaxis calculation. Note: x1,y1 are from SS2 and x2, y2 from SS1!!"""
    #stereo params
    """float B = distance between cameras in cm
    float f = focal length camera in cm
    float px = number of px per cm
    float  z1 = location of refractive surface in cm
    float kappa = ratio of refractive indices n_air/n_H2O = 1./1.33 """
    B, f, px, z1, kappa = 8.3, 0.304, 112*10**-6, 68, 1.33
    # correct pixel size due to image rescaling
    sensorx, sensory = paramDict['sensorPxX'], paramDict['sensorPxY']
    # image dimensions from params
    correctionFactory = sensory/paramDict['imHeight']
    # multiply px value with correction to account for actually 'larger' pixels due to rescaling
    pxNew = px*correctionFactory
     # rotate coordinates to account for image issues
    
    x1, y1 = rotatePoints(x1,y1,paramDict, flag = 'SS2')
    
    x2, y2 = rotatePoints(x2,y2,paramDict, flag = 'SS1')
    # center points
    y1 = pxNew*(paramDict['imHeight']/2.-y1)
    y2 = pxNew*(paramDict['imHeight']/2.-y2)
    x1 = pxNew*(paramDict['imWidth']/2.-x1)
    x2  = pxNew*(paramDict['imWidth']/2.-x2)
    # calculate depth from disparity. We use small-angle approximation for z due to diffraction
    Z = B*f/(y1-y2)*kappa
    X = -(x1+x2)/2.*Z/f # fix x axis
    Y = (y1+y2)/2.*Z/f
    
    return np.vstack([X,Y,Z])

#=============================================================================
#
#       image analysis functions
#
#=============================================================================      
def rotationAngle(bgIm, paramDict):
    # find a rotation angle to correct image skew.
    nC =2
    plt.figure('click on two points that lie on top of the tank', figsize=(16,12))
    plt.subplot(111)
    plt.imshow(bgIm)
    locs1 = np.array(plt.ginput(nC, timeout=0))
    plt.close()
    x, y = locs1[:,0], locs1[:,1]
    theta = np.arctan(np.diff(y)/np.diff(x))[0]
    bg1n = rotate(bgIm, (theta)*360/(np.pi*2))
    # rotate a set of points
    c, s = np.cos(theta), np.sin(theta)
    print c, s
    T = np.matrix(np.array([paramDict['imWidth']/2., paramDict['imHeight']/2.]))
    R = np.matrix(np.array([[c,s], [-s, c]]))
   
    xn, yn = np.dot(R, np.vstack([x,y])-T)+T
    
    plt.subplot(111)
    plt.imshow(bg1n)
    plt.plot(locs1[:,0], locs1[:,1], 'ro')
    plt.plot(xn, yn, 'ko')
    plt.show()
    return theta
  
def findObjects(trackIm0, paramDict, flag):
    """find objects after image segmentation etc and return pertinent parameters.""" 
    # invert image and get rid of some smaller noisy parts by gaussian smoothing
    # invert image and get rid of some smaller noisy parts by gaussian smoothing
    trackIm = np.abs(trackIm0)
    width, height = trackIm.shape
    trackIm = gaussian(trackIm, sigma = 7)
    # block out strange area in SS2
    if flag =='SS2':
        trackIm[70:145,530:610] = 0
    #trackIm = median(trackIm, disk(10))
    #thresh = np.percentile(trackIm,[100-100/width*height])
    # apply threshold
    thresh = threshold_yen(trackIm)
    
    bw = opening(trackIm > thresh, square(5))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    # define regions
    rprops =  regionprops(label_image, trackIm)
    #
    
    # show tracking
#    plt.subplot(221)
#    plt.imshow(trackIm)
#    plt.subplot(222)
#    plt.imshow(bw)
#    
#    plt.subplot(223)
#    plt.imshow(label_image)
    locs = []
    for region in rprops:
        yob, xob = region.centroid
        area = region.area
        if area>paramDict['starsize']:
            locs.append([xob, yob])
    return np.array(locs)

def subtractBg(tmpIm, meanBrightness, bgs):
    """subtract either night or day background from image depending on brightness."""
    if np.mean(tmpIm) >= meanBrightness:
            return tmpIm - bgs[0]
    else:
            return tmpIm - bgs[1]

def findMatchDistance(dist, N):
    """given a distance matrix, find the entries and values of the first N minima."""
    results = []
    for n in range(N):
        loc, value = np.unravel_index(dist.argmin(), dist.shape), np.min(dist)
        results.append([loc, value])
        dist[loc[0]] = np.inf
        dist[:,loc[1]] = np.inf
    return results

def filterCoords(linked3D, paramDict):
    """remove out of tank coordinates."""
    xmin, xmax, ymin,ymax, zmin, zmax = -45, 45, -30, 30, 60, 120
    for time, coords in enumerate(linked3D):
        for n in range(paramDict['NStars']):
#            if np.sum(coords[n, 4:])==0:
#                linked3D[time, n,:] = np.repeat(np.nan, 7)
                #check if in tank
            if xmin <= coords[n, 4] <=xmax and  ymin <= coords[n, 5] <=ymax and  zmin <= coords[n, 6] <=zmax:
                continue
            else:
                linked3D[time, n,:] = np.repeat(np.nan, 7)
    return linked3D
#=============================================================================
#
#       User input
#
#============================================================================= 
def initializePos(trackIm, paramDict, linked, ss2Im = None):
    # get starting locations for all stars in the image
    
    plt.figure('Click on each star in the image. We expect {} click(s) in each image.'.format(paramDict['NStars']), figsize=(18,6))
    plt.subplot(121)
    plt.imshow(trackIm)
    for n in range(paramDict['NStars']):
        plt.plot(linked[:,n,0], linked[:,n,1], 'o', label='Seastar {}'.format(n))
    h, w = paramDict['imHeight']/2, paramDict['imWidth']/2
    plt.axhline(h, color='w')
    plt.axvline(w, color='w')
    plt.legend()
    if ss2Im is not None:
        plt.subplot(122)
        plt.imshow(ss2Im)
    locs = plt.ginput(paramDict['NStars'], timeout=0)
    
    plt.close()
    
    # note: ginput returns x,y but images are typically y,x in array axes
    return locs
    
def initializePos3D1(ss1Im, ss2Im, paramDict, coords1, coords2, lastLocs = None):
    # get starting locations for all stars in the image
    plt.figure('Click on the corresponding star in each image. We expect 1 click(s) in each image.', figsize=(18,6))
    plt.subplot(121)
    plt.imshow(ss1Im)
    if len(coords1):
        plt.scatter(coords1[:,0], coords1[:,1], color='w')
    if lastLocs is not None:
        colors = ['r', 'orange']
        for n in range(paramDict['NStars']):
            plt.scatter(lastLocs[:,n,0], lastLocs[:,n,1], s=1, color=colors[n], label=n)
            plt.scatter(lastLocs[-1,n,0], lastLocs[-1,n,1], s=5, color=colors[n])
            plt.legend()
    plt.subplot(122)
    plt.imshow(ss2Im)
    if len(coords2):
        plt.scatter(coords2[:,0], coords2[:,1], color='w')
    if lastLocs is not None:
        colors = ['r', 'orange']
        for n in range(paramDict['NStars']):
            plt.scatter(lastLocs[:,n,2], lastLocs[:,n,3], s=1, color=colors[n], label=n)
            plt.scatter(lastLocs[-1,n,2], lastLocs[-1,n,3], s=5, color=colors[n])
            plt.legend()
    locs = plt.ginput(2, timeout=0)
    #plt.pause(1)
    plt.close()
    x2,y2 = locs[0]
    x1,y1 = locs[1]
    # calculate 3D location too
    X,Y,Z = calculate3DPoint(x1, y1, x2, y2, paramDict)
    # note: ginput returns x,y but images are typically y,x in array axes
    return x2,y2,x1,y1, X,Y,Z
    
    
    
def initializePos3D(ss1Im, ss2Im, paramDict, coords1, coords2, lastLocs = None, star=0):
    fig = plt.figure('Star number {}. Click on the corresponding star in each image.'.format(star), figsize=(24,12))
    plt.subplot(121)
    plt.imshow(ss1Im)
    class Clicker:
        def __init__(self, fig):
            
            self.locs = []
            self.nothing = False
            self.cid = fig.canvas.mpl_connect('button_press_event', self)
            self.cid2 = fig.canvas.mpl_connect('key_press_event', self)
            self.key = None
    
        def __call__(self, event):
            print('click', event)
            
            if event.key is not None:
                ### allow only two clicks
                fig.canvas.mpl_disconnect(self.cid)
                fig.canvas.mpl_disconnect(self.cid2)
                plt.close(fig)
                print 'Done'
            elif event.inaxes:
                #### check if points are within images
                self.locs.append((event.xdata, event.ydata))

    clicker = Clicker(fig)
    if len(coords1):
        plt.scatter(coords1[:,0], coords1[:,1], color='w', alpha=0.5)
    if lastLocs is not None:
        colors = ['r', 'orange']
        for n in range(paramDict['NStars']):
            plt.scatter(lastLocs[:,n,0], lastLocs[:,n,1], s=1, color=colors[n], label=n)
            plt.scatter(lastLocs[-1,n,0], lastLocs[-1,n,1], s=15,marker='*', color=colors[n])
            plt.legend()
    plt.subplot(122)
    plt.imshow(ss2Im)
    if len(coords2):
        plt.scatter(coords2[:,0], coords2[:,1], color='w')
    if lastLocs is not None:
        colors = ['r', 'orange']
        for n in range(paramDict['NStars']):
            plt.scatter(lastLocs[:,n,2], lastLocs[:,n,3], s=1, color=colors[n], label=n)
            plt.scatter(lastLocs[-1,n,2], lastLocs[-1,n,3], s=15,marker='*', color=colors[n], label='Z={:.1f}'.format(lastLocs[-1,n,-2]))
            plt.legend()
    plt.tight_layout()
    plt.show()
    locs = clicker.locs
    if len(locs)<2:
        # just grab the last known location
        # if confirmed by enter, accept location as good and set flag to 1
        x2,y2,x1,y1, X,Y,Z, _ = lastLocs[-1,star]
        if clicker.key =='enter':
            return x2,y2,x1,y1, X,Y,Z, 1
        return x2,y2,x1,y1, X,Y,Z, -1
    
    x2,y2 = locs[0]
    x1,y1 = locs[1]
    
    # calculate 3D location too
    X,Y,Z = calculate3DPoint(x1, y1, x2, y2, paramDict)
    # note: ginput returns x,y but images are typically y,x in array axes
    return x2,y2,x1,y1, X,Y,Z, -1
#=============================================================================
#
#       macros
#
#============================================================================= 
        
def calculateBackground(impath, condition, analysisPath, paramDict, flag, show_figs = True):
    """Calculate background images by median. First look at overall pixel change to get daylight/nightlight, 
    then use this info to calculate day and night backgrounds."""
    smallSet = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['bgstep'])
    imgs = io.ImageCollection(smallSet, load_func = imread_convert, flag=flag, rgb = paramDict['rgb'])
    imgs = io.concatenate_images(imgs).astype(np.float)
    print "No. images: {} \n Height x Width (px): {}x{}".format(*imgs.shape)

    bgIm = np.median(imgs, axis = 0)
    
    brightness = np.mean(imgs, axis = (1,2))  
    
    time = np.arange(paramDict['start'],len(smallSet))*paramDict['bgstep']/paramDict['framerate']# time in minutes
    
    # refine by calculating day and night backgrounds
    meanBrightness = np.mean(brightness)
    binBrightness = brightness >= meanBrightness
    dayBgIm = np.median(imgs[binBrightness], axis = 0)
    nightBgIm = np.median(imgs[~binBrightness], axis = 0)
    
    # create a list of background images and subtract the corresponding ones
    fullBg = np.zeros(imgs.shape)
    for i in range(len(imgs)):
        if brightness[i] >= meanBrightness:
            fullBg[i] = dayBgIm
        else:
             fullBg[i] = nightBgIm
    
    # recalculate activity on subtracted images
    corrImgs = imgs-fullBg
    #corrImgs = np.where(corrImgs>0, corrImgs,1)
    
    nactivity = np.mean(np.abs(np.diff(corrImgs, axis=0)),axis =(1,2))
    
    io.imsave(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,flag)), dayBgIm)
    io.imsave(os.path.join(analysisPath, 'BG_Night_{}_{}.jpg'.format(condition,flag)), nightBgIm)
    
    np.savetxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition,flag)), np.vstack([time[:-1], nactivity, brightness[:-1]]).T, header = "time, activity, brightness")


def detectStars(impath, condition, analysisPath, paramDict, dayBgIm, nightBgIm, meanBrightness, flag, show_figs= False):
    """traditional threshold-segment routine to detect specific sized objects after background subtraction."""
    allImFiles = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['step'])
    tracks = []
    
    for imIndex, imFile in enumerate(allImFiles):
        if imIndex%100==0:
            print 'Finding the stars in frame ', imIndex
        # read image and subtract day/night Background
        trackIm = subtractBg(imread_convert(imFile, flag, paramDict['rgb']), meanBrightness, [dayBgIm, nightBgIm])
        # do object detection - starLocs is in y,x order == plot in image as plt.plot(starLocs[1], starLocs[0])
        starLocs = findObjects(trackIm, paramDict, flag)
        
        tracks.append(starLocs)
#        plt.imshow(trackIm)
#        for loc in starLocs:
#            plt.plot(loc[0], loc[1], 'ro')
#        plt.show()
    # write putative star locations to file
    frameIndices = np.arange(paramDict['start'],paramDict['end'],paramDict['step'])
    
    with open(os.path.join(analysisPath, 'Tracks_{}_{}.txt'.format(condition, flag)), 'w') as f:
        f.write("# x,y Locations of stars in y,x pairs \n")
        for findex, frames in enumerate(tracks):
            f.write('{} '.format(frameIndices[findex]))
            for coords in frames:
                f.write("{} {} ".format(*coords))
            f.write('\n')
    return tracks


def linkingTrajectories3D(impath, impath2,  analysisPath, tracks, paramDict,condition,):
    """use 3D coordinates to find closest paths."""
    # tank coordinates in cm
    xmin, xmax, ymin,ymax, zmin, zmax = -45, 45, -30, 30, 60, 130
    frames = np.array(tracks.keys(), dtype = int)
    #frames = np.arange(paramDict['start'],paramDict['end'], paramDict['step'])
    allImFiles = loadSubset(impath , extension=paramDict['ext'])[frames]
    allImFiles2 = loadSubset(impath2 , extension=paramDict['ext'])[frames]
    linked3D = np.ones((len(allImFiles),paramDict['NStars'], 8))*np.nan
    matches = np.ones((len(allImFiles),2))*np.nan
    for n in range(paramDict['NStars']):
        nLost = 1
        for imIndex, imFile in enumerate(allImFiles):
            cIndex = frames[imIndex]
            #load all identified objects in SS1 and SS2
            coords1 = np.reshape(tracks[cIndex][0], (-1,2))
            coords2 = np.reshape(tracks[cIndex][1], (-1,2))
            # remove previous matches
            if n==1 and np.isfinite(matches[imIndex][0]):
                # pop from tracks
                coords1[int(matches[imIndex][0])] = [np.nan, np.nan]
                coords2[int(matches[imIndex][1])] = [np.nan, np.nan]
              # some progress reporting
            if imIndex%1==0:
                print 'Linking frame ', imIndex
            
            # initialize the first star
            if imIndex ==0:
                # user input start coordinates
                ss1Im = imread_convert(allImFiles[imIndex], 'SS1', paramDict['rgb'])
                ss2Im = imread_convert(allImFiles2[imIndex], 'SS2', paramDict['rgb'])
                x1,y1,x2,y2, X,Y,Z,_ = initializePos3D(ss1Im, ss2Im, paramDict, coords1, coords2, star=n)
                
                #linked3D[imIndex,n] = x1,y1,x2,y2, X,Y,Z
                currLoc =  x1,y1,x2,y2, X,Y,Z, -1
                nLost = 1
                
            if  nLost > paramDict['lostFrames']:
                  # user input start coordinates
                ss1Im = imread_convert(allImFiles[imIndex], 'SS1', paramDict['rgb'])
                ss2Im = imread_convert(allImFiles2[imIndex], 'SS2', paramDict['rgb'])
                lastLocs = linked3D[np.where(np.isfinite(linked3D[:,n,0]))[0][-200:]]
                if len(lastLocs) ==0:
                    lastLocs=None
                click = initializePos3D(ss1Im, ss2Im, paramDict, coords1, coords2, lastLocs, star=n)
                # if previous flag bad, leave it that way
                if linked3D[imIndex,0, -1] == -1:
                    linked3D[imIndex,n] =  click
                    linked3D[imIndex,n, -1] =  -1
                else:
                    linked3D[imIndex,n] = click
                currLoc = click
                # if we have a good quality points, keep clicking
                if click[-1] != 1:
                    nLost = -5
            # calculate all putative 3d points
            locations = []
            ids = []
            for xi, (x,y) in enumerate(coords1):
                for di,(d,e) in enumerate(coords2):
                    X,Y,Z = calculate3DPoint(d,e,x,y, paramDict)
                    locations.append([x,y,d,e,X,Y,Z,1])
                    ids.append([xi,di])
            # calculate distance of current to putative locations
                # get latest coordinates
            x1,y1,x2,y2,X0,Y0,Z0,_ =  currLoc
            dist = np.ones(len(locations))*np.nan
            for kij in range(len(locations)):
                xt,_,_,_,X,Y,Z, _ =  locations[kij]
                if xmin <= X <=xmax and  ymin <= Y <=ymax and  zmin <= Z <=zmax:
                    dist[kij] = np.sqrt((X-X0)**2+(Y-Y0)**2+(Z-Z0)**2)
                else:
                    print X,Y,Z
            # ignore frames where we don't have any valid matches anyway
            if len(dist)==0 or np.all(np.isnan(dist)):
                continue

            # identify closest point in 3D
            match = np.nanargmin(dist)
            
            #print dist, coords1[match][0]-x1
            if dist[match] < (paramDict['3DMaxDist']) and np.abs(locations[match][0]-x1)<paramDict['xTolerance']:
                # if previous flag bad, leave it that way
                if linked3D[imIndex,0, -1
                
                ] == -1:
                    linked3D[imIndex,n] =  locations[match]
                    linked3D[imIndex,n, -1] =  -1
                else:
                    # append to coordinate
                    linked3D[imIndex,n] =  locations[match]
                    #update current position in 3D and set flag to indicate good track
                currLoc = linked3D[imIndex,n]
                # reset lost frames
                nLost = 1
                # save match to remove this pair from consideration for the other star
                matches[imIndex] = ids[match]
                
            else:
                print 'No', dist[match],  np.abs(locations[match][0]-x1)
                nLost += 1
                
    #linked3D = filterCoords(linked3D, paramDict)
    c = ['#001f3f', '#85144b']
    for n in range(paramDict['NStars']):
        plt.subplot(paramDict['NStars'],1,n+1)
        plt.plot(linked3D[:,n,0], linked3D[:,n,1], 'o', color=c[n])
        plt.plot(linked3D[:,n,2], linked3D[:,n,3], 'o', color=c[n], alpha=0.05)
        plt.show()
    # filter out 0 points
    plt.subplot(311)
    plt.imshow(ss1Im)
    for n in range(paramDict['NStars']):
        plt.scatter(linked3D[:,n,0], linked3D[:,n,1], s=1)
        plt.subplot(312)
        plt.plot(linked3D[:,n,0])
        plt.plot(linked3D[:,n,2])
        plt.subplot(313)
        plt.plot(linked3D[:,n,1])
        plt.plot(linked3D[:,n,3])
    plt.show()
    plt.savetxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)), np.reshape(linked3D, (linked3D.shape[0], -1)), header = "#xs1, ys1, xs2, ys2, X,Y,Z")
    return linked3D
                

    
#=============================================================================
#
#       Main code
#
#=============================================================================



def main():
    imPath = '/media/monika/MyPassport/Ns/{}{}Solo'
    imPath = '/media/monika/MyPassport/Ns/{}{}Ns'
    analysisPath = '/media/monika/MyPassport/Ns/Analysis/'
##########################################
#    imPath = '/media/monika/MyPassport/Os/{}{}Solo'
    imPath = '/media/monika/MyPassport/Os/{}{}Os'
    analysisPath = '/media/monika/MyPassport/Os/Analysis/'
##############################################
#    imPath = '/media/monika/MyPassport/Qs/{}{}'
#    #imPath = '/media/monika/MyPassport/Qs/{}{}Os'
#    analysisPath = '/media/monika/MyPassport/Qs/Analysis/'
##############################################
#    imPath = '/media/monika/MyPassport/Ps/{}{}'
#    #imPath = '/media/monika/MyPassport/Ps/{}{}Ps'
#    analysisPath = '/media/monika/MyPassport/Ps/Analysis/'
    
   
    bgCalc = 0
    tracking = 0
    linking = 1#True
    # which movie to analyze
    condition = 'Both'
    # save parameters
    # calibrate the rotation data
    
    # read the stored data and background images
    if bgCalc:
        for flag in ['SS1', 'SS2']:
            paramDict = {'start':0,
             'end':12045,
             'step':1,
             'bgstep': 50,
             'ext' : ".jpg",
            'rgb' : True,
            'framerate':3., #frames per minute
            'starsize':0,
            'NStars': 1, # How many stars should we look for,
            'minDistance':5, # in pixels per frame,
            'imHeight': 616,# actual image size (we rescale before saving)
            'imWidth': 820,
            'sensorPxX':3280,# maximum resolution of sensor
            'sensorPxY':2464,
            'lostFrames':10, # how many frames can a star not be visible until we reinitialize 
            'xTolerance': 20,#how much jiggle in x axis is acceptable between cameras in px (measured 16 px)
            '3DMaxDist': 25,#distance stars move per frame in cm
            'rotateSS1':0, #rotation angle in each camera
            'rotateSS2':0 #rotation angle
            }
            #paramDict =  readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
            calculateBackground(imPath.format(flag,condition),condition, analysisPath, paramDict, flag = flag, show_figs = True)
            # rotate images
            bgIm  = imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,flag))).astype(np.float)
            paramDict['rotate{}'.format(flag)]= rotationAngle(bgIm, paramDict)            
            # save parameters
            writePars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)), paramDict)
            # write status
            writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                        action = "BG detection {} frames {} - {}".format(flag, paramDict['start'], paramDict['end']))
   
    if tracking:
        for flag in ['SS1','SS2']:
            # load params
            paramDict =  readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
            # load existing background
            dayBgIm, nightBgIm = imread_convert(os.path.join(analysisPath, 'BG_Day_{}_{}.jpg'.format(condition,flag))).astype(np.float)\
            , imread_convert(os.path.join(analysisPath, 'BG_Night_{}_{}.jpg'.format(condition, flag))).astype(np.float)
            
            time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}_{}.txt'.format(condition, flag)), unpack=True)
            meanBrightness = np.mean(brightness)
            # run object detection
            detectStars(imPath.format(flag, condition), condition, analysisPath, paramDict, dayBgIm, nightBgIm, meanBrightness, flag, show_figs= False)
            # write status
            writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                        action = "Tracking {} frames {} - {}".format(flag, paramDict['start'], paramDict['end']))

    
    if linking:
        
        # load params
        paramDict =  readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
        print repr(paramDict)
        plt.waitforbuttonpress()
        # load existing tracks
        tracks = loadTracks(analysisPath, condition)
        # link simultaneously in 3D
        linkingTrajectories3D(imPath.format('SS1', condition),imPath.format('SS2', condition), analysisPath, tracks,  paramDict, condition)
      
        
        
        
        writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                        action = "3D linking frames {} - {}".format(paramDict['start'], paramDict['end']))

if __name__ == "__main__":
    main()