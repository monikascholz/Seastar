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
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, medfilt
from skimage.transform import warp, rotate
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
  
def findObjects(trackIm, paramDict):
    """find objects after image segmentation etc and return pertinent parameters.""" 
    # invert image and get rid of some smaller noisy parts by gaussian smoothing
    trackIm = np.abs(trackIm)
    trackIm = gaussian(trackIm, sigma = 5)
    
    # apply threshold
    thresh = threshold_yen(trackIm)
    bw = opening(trackIm > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    # define regions
    rprops =  regionprops(label_image, trackIm)
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
def initializePos(trackIm, paramDict, linked):
    # get starting locations for all stars in the image
    
    plt.figure('Click on each star in the image. We expect {} click(s) in each image.'.format(paramDict['NStars']), figsize=(12,8))
    plt.subplot(111)
    plt.imshow(trackIm)
    for n in range(paramDict['NStars']):
        plt.plot(linked[:,n,0], linked[:,n,1], 'o', label='Seastar {}'.format(n))
    h, w = paramDict['imHeight']/2, paramDict['imWidth']/2
    plt.axhline(h, color='w')
    plt.axvline(w, color='w')
    plt.legend()
    locs = plt.ginput(paramDict['NStars'], timeout=0)
    plt.close()
    # note: ginput returns x,y but images are typically y,x in array axes
    return locs
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
    activity = np.mean(np.abs(np.diff(imgs-bgIm, axis=0)),axis =(1,2))
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
    if show_figs:
        # calculate rough estimates
        plt.figure('Simple median background')
        plt.subplot(221)
        io.imshow(bgIm)
        plt.subplot(222)
        io.imshow(np.abs(corrImgs[0]), show_cbar=True)
        plt.subplot(223)
        plt.plot(time, brightness)
        plt.step(time, binBrightness*(np.max(brightness) - np.min(brightness)) + np.min(brightness))
    
        plt.ylabel('Mean Brightness')
        plt.xlabel('Time (min)')
        plt.subplot(224)
        plt.plot(time[:-1], activity)
        plt.xlabel('Time (min)')
        plt.ylabel('Acitvity')
        plt.show()
        
        
        plt.figure('Day and night median backgrounds')
        plt.subplot(221)
        plt.title('Daylight background')
        plt.imshow(dayBgIm)
        plt.subplot(222)
        plt.title('Night background')
        plt.imshow(nightBgIm)
        plt.subplot(223)
        plt.title('Subtracted day')
        plt.imshow(corrImgs[0])
        plt.subplot(224)
        plt.title('Subtracted night')
        plt.imshow(corrImgs[np.where(brightness<meanBrightness)[0][0]])
        plt.show()
    
        plt.figure('Activity and brightness')
    
        plt.subplot(221)
        plt.plot(time, brightness)
        plt.ylabel('Mean Brightness')
        plt.xlabel('Time (min)')
        
        plt.subplot(222)
        plt.plot(time[:-1], nactivity)
        plt.step(time, binBrightness*(np.max(nactivity) - np.min(nactivity)) + np.min(nactivity))
        plt.xlabel('Time (min)')
        plt.ylabel('Acitvity')
        
        plt.subplot(223)
        plt.scatter(brightness[:-1], nactivity)
        plt.axvline(meanBrightness)
        plt.xlabel('Brightness')
        plt.ylabel('Acitvity')
        
        plt.subplot(224)
        plt.scatter(np.abs(np.diff(brightness)), nactivity)
        #plt.axvline(meanBrightness)
        plt.xlabel('Change in Brightness')
        plt.ylabel('Acitvity')
        
        plt.show()
    
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
        starLocs = findObjects(trackIm, paramDict)
        
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


def linkingTrajectories3D(impath, condition, analysisPath, tracks, paramDict, linked):
    """use previously linked first camera coordinate to link to second."""
    frames = np.array(tracks.keys(), dtype = int)
    allImFiles = loadSubset(impath , extension=paramDict['ext'])[frames]
    linked3D = np.ones((len(allImFiles),paramDict['NStars'], 7))*np.nan
    curr3D = np.zeros((paramDict['NStars'], 3))
    nLost = np.ones(paramDict['NStars'])
    for imIndex, imFile in enumerate(allImFiles):
        cIndex = frames[imIndex]
        coords = np.reshape(tracks[cIndex][1], (-1,2))
        if len(coords) == 0:
            continue

        if imIndex == 0 or np.any(nLost>paramDict['lostFrames']):
            
            # initialize 3D location
            firstIm = imread_convert(allImFiles[imIndex], 'SS2', paramDict['rgb'])
            
            currPos = initializePos(firstIm, paramDict, linked[imIndex:imIndex+10])
            
            # Calculate original 3d location
            for n in range(paramDict['NStars']):
                curr3D[n] = np.squeeze(calculate3DPoint(currPos[n][0], currPos[n][1],linked[imIndex][n][0], linked[imIndex][n][1], paramDict))
            nLost = np.ones(paramDict['NStars'])
            #print curr3D, 0/0
        # calculate all 3D coordinates and putative 3D distance
        for n in range(paramDict['NStars']):
            x1, y1 = linked[imIndex][n]
            linked3D[imIndex,n, :2] = x1, y1
            if np.isnan(x1):
                print 'hello'
            X0,Y0,Z0 = curr3D[n]
            dist = np.zeros(len(coords))
            #threedim = np.zeros((len(coords), 3))
            threedim = np.array(calculate3DPoint(coords[:,0], coords[:,1],x1, y1, paramDict)).T
            
            for kij in range(len(coords)):
                X,Y,Z = threedim[kij]
                #print X,Y,Z
                dist[kij] = np.sqrt((X-X0)**2+(Y-Y0)**2+(Z-Z0)**2)
            
            #print 0/0
            # identify closest point in 3D
            match = np.argmin(dist)
            
            if dist[match] < (paramDict['3DMaxDist']*nLost[n]) and np.abs(coords[match][0]-x1)<paramDict['xTolerance']:
                # append to coordinate
                X,Y,Z = threedim[match]
                linked3D[imIndex,n] = x1, y1, coords[match][0],coords[match][1], X,Y,Z
                #update current position in 3D
                curr3D[n] = threedim[match]
                nLost[n] = 1
            else:
                nLost[n]+=1
                
    #linked3D = filterCoords(linked3D, paramDict)
    c = ['#001f3f', '#85144b']
    for n in range(paramDict['NStars']):
        plt.subplot(paramDict['NStars'],1,n+1)
        plt.plot(linked3D[:,n,0], linked3D[:,n,1], 'o', color=c[n])
        plt.plot(linked3D[:,n,2], linked3D[:,n,3], 'o', color=c[n], alpha=0.05)
        plt.show()
    # filter out 0 points
    plt.subplot(311)
    plt.imshow(firstIm)
    for n in range(paramDict['NStars']):
        plt.plot(linked3D[:,n,0], linked[:,n,1], 'o')
        plt.subplot(312)
        plt.plot(linked3D[:,n,0])
        plt.plot(linked3D[:,n,2])
        plt.subplot(313)
        plt.plot(linked3D[:,n,1])
        plt.plot(linked3D[:,n,3])
    plt.show()
    plt.savetxt(os.path.join(analysisPath, 'linked3d_{}.txt'.format(condition)), np.reshape(linked3D, (linked3D.shape[0], -1)), header = "#xs1, ys1, xs2, ys2, X,Y,Z")
    return linked3D
                
def linkingTrajectories2D(impath, analysisPath, tracks, paramDict, condition):
    """link the objects together using user-defined start coordinates and closest match."""
    frames = np.array(tracks.keys(), dtype = int)
    allImFiles = loadSubset(impath , extension=paramDict['ext'])[frames]
    
    # empty output array for 2-d and 3D coordinates
    linked = np.ones((len(allImFiles),paramDict['NStars'], 2))*np.nan
    nLost = np.ones(paramDict['NStars'])
    for imIndex, imFile in enumerate(allImFiles):
        # track which coordinate
        cIndex = frames[imIndex]
        #print nLost
        if imIndex ==0 or np.any(nLost>paramDict['lostFrames']):
            # fx for now
            #currPos = [(565.30524344569289, 356.94319600499381), (578.28901373283384, 269.05305867665425)]
            # user input start coordinates
            firstIm = imread_convert(allImFiles[imIndex], 'SS1', paramDict['rgb'])
            #print linked.shape
            #print linked[imIndex-50:imIndex,:]
            currPos = initializePos(firstIm, paramDict, linked[imIndex-10:imIndex])
            nLost = np.ones(paramDict['NStars'])
        if imIndex%10==0:
            print 'Linking frame ', imIndex
            # dealing with too few stars
        if len(tracks[cIndex][0]) == 0:
            nLost += 1
            continue
       
        coords = np.reshape(tracks[cIndex][0], (-1,2)) # reshape to recreate coordinate pairs
        # first, calculate distance metric for all pairs of stars in SS1
        dist = np.zeros((paramDict['NStars'], len(coords)))
        for n in range(paramDict['NStars']):
            for lindex, coord in enumerate(coords):
                dist[n, lindex] = np.sqrt(np.sum((currPos[n] - coord)**2))
        # identify matches
        
        matches = findMatchDistance(dist, paramDict['NStars'])
        
        for loc, value in matches:
            nStar, coordStar = loc
            if value < paramDict['minDistance']*nLost[nStar]:
                # append to coordinate
                linked[imIndex,nStar, :2] = coords[coordStar]
                #update current position in SS1 camera
                currPos[nStar] = coords[coordStar]
                nLost[nStar] = 1
            else:
                nLost[nStar]+=1
    
    # interpolate 2d trajectories
    for n in range(paramDict['NStars']):
        mask = np.isnan(linked[:,n,0])
        linked[:,n,0][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked[:,n,0][~mask])
        linked[:,n,1][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),linked[:,n,1][~mask])
        
    plt.figure()
    plt.subplot(311)
    plt.imshow(firstIm)
    for n in range(paramDict['NStars']):
        plt.plot(linked[:,n,0], linked[:,n,1], 'o')
        plt.subplot(312)
        plt.plot(linked[:,n,0])
        plt.subplot(313)
        plt.plot(linked[:,n,1])
    plt.show()
    plt.savetxt(os.path.join(analysisPath, 'linked2d_{}.txt'.format(condition)), np.reshape(linked, (linked.shape[0], -1)), header = "#xs1, ys1, xs2, ys2")
    return linked
    
#=============================================================================
#
#       Main code
#
#=============================================================================



def main():
    imPath = 'MS_20180205_N2_1_100mB'
    
    analysisPath = 'Analysis/'
    
    paramDict = {'start':0,
                 'end':None,
                 'step':1,
                 'bgstep': 140,
                 'ext' : ".jpg",
                'rgb' : True,
                'framerate':3., #frames per minute
                'starsize':0,
                'NStars': 1, # How many stars should we look for,
                'minDistance':15, # in pixels per frame,
                'imHeight': 616,# actual image size (we rescale before saving)
                'imWidth': 820,
                'sensorPxX':3280,# maximum resolution of sensor
                'sensorPxY':2464,
                'lostFrames':20, # how many frames can a star not be visible until we reinitialize 
                'xTolerance': 25,#how much jiggle in x axis is acceptable between cameras in px (measured 16 px)
                '3DMaxDist': 5,#distance stars move per frame in cm
                'rotateSS1':0, #rotation angle in each camera
                'rotateSS2':0 #rotation angle
    }
    startAnalysis = 1
    bgCalc = 1
    tracking = 1
    linking = 0#True
    parallax = 1
    # which movie to analyze
    condition = 'MS_20180205_N2_1_100mB'
    flag = ''
    # save parameters
    if startAnalysis:
       
       writePars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)), paramDict)
    
    # read the stored data and background images
    if bgCalc:
        
        paramDict =  readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))
        calculateBackground(imPath.format(flag,condition),condition, analysisPath, paramDict, flag = flag, show_figs = True)
        # write status
        writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                    action = "BG detection {} frames {} - {}".format(flag, paramDict['start'], paramDict['end']))
   
    if tracking:
        for flag in ['SS1', 'SS2']:
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
        # load existing tracks
        tracks = loadTracks(analysisPath, condition)
        linkingTrajectories2D(imPath.format('SS1', condition), analysisPath, tracks,  paramDict, condition)
        # write status
        writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                        action = "2D Linking frames {} - {}".format(paramDict['start'], paramDict['end']))
    
    if parallax:
        # load params
        paramDict =  readPars(os.path.join(analysisPath, '{}_pars.txt'.format(condition)))

        linked = np.reshape(np.loadtxt(os.path.join(analysisPath, 'linked2d_{}.txt'.format(condition))), (-1,paramDict['NStars'], 2))
        tracks = loadTracks(analysisPath, condition)
        linkingTrajectories3D(imPath.format('SS2', condition), condition, analysisPath, tracks, paramDict, linked)
        # write status
        writeStatus(os.path.join(analysisPath, '{}_status.txt'.format(condition)), \
                        action = "3D linking frames {} - {}".format(paramDict['start'], paramDict['end']))

if __name__ == "__main__":
    main()