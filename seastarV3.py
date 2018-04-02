# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:08:53 2017
Track Seastars in images.
@author: monika
"""

import skimage

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

def calculateBackground(impath, step, flag, show_figs = True, rgb = True):
    """Calculate background images by median. First look at overall pixel change to get daylight/nightlight, 
    then use this info to calculate day and night backgrounds."""
    smallSet = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = step)
    imgs = io.ImageCollection(smallSet, load_func = imread_convert, flag=flag, rgb = rgb)
    imgs = io.concatenate_images(imgs).astype(np.float)
    print "No. images: {} \n Height x Width (px): {}x{}".format(*imgs.shape)

    bgIm = np.median(imgs, axis = 0)
    
    brightness = np.mean(imgs, axis = (1,2))  
    activity = np.mean(np.abs(np.diff(imgs-bgIm, axis=0)),axis =(1,2))
    time = np.arange(paramDict['start'],len(smallSet))*step/paramDict['framerate']# time in minutes
    
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
    
    io.imsave(os.path.join(analysisPath, 'BG_Day_{}.jpg'.format(flag)), dayBgIm)
    io.imsave(os.path.join(analysisPath, 'BG_Night_{}.jpg'.format(flag)), nightBgIm)
    
    np.savetxt(os.path.join(analysisPath, 'BgData_{}.txt'.format(flag)), np.vstack([time[:-1], nactivity, brightness[:-1]]).T, header = "time, activity, brightness")
    

def natural_sort(liste):
    """Natural sort to have frames in right order."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(liste, key = alphanum_key)


def loadSubset(impath, extension, start = 0, end = None, step = 1):
    """load images with file extension extension following the pattern given by tuple.
    example: loadSubset('.', 'png', start=0, end=100, step =4) loads every 4th image in the directory up to the hundredth image."""
    imFileList = natural_sort([os.path.join(impath,f) for f in os.listdir(impath) if f.endswith(extension)])[start:end:step]
    return imFileList

def imread_convert(f, flag='SS1', rgb=False):
    if rgb:
        if flag=='SS2':
            return img_as_float(io.imread(f))[::-1,::-1,0]
        return img_as_float(io.imread(f))[:,:,0]
    else:
        if flag=='SS2':
            return img_as_float(io.imread(f, as_grey=True))[::-1,::-1]
        return img_as_float(io.imread(f, as_grey=True))

def calculate3DPoint(x1, y1, x2, y2, paramDict):
    """single point parallaxis calculation."""
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
    # multiply px value with correction to account for actually 'larger' pixels
    pxNew = px*correctionFactory
    
    y1 = pxNew*(paramDict['imHeight']/2.-y1)
    y2 = pxNew*(paramDict['imHeight']/2.-y2)
    x1 = pxNew*(paramDict['imWidth']/2.-x1)
    x2  = pxNew*(paramDict['imWidth']/2.-x2)

    # calculate depth from disparity. We use small-angle approximation for z due to diffraction
    Z = -B*f/(y1-y2)*kappa
    X = (x1+x2)/2.*Z/f
    Y = (y1+y2)/2.*Z/f
    return X,Y,Z
    
def calculate2DPoint(X,Y,Z, paramDict):
    """single point reverse parallaxis calculation."""
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
    # multiply px value with correction to account for actually 'larger' pixels
    pxNew = px*correctionFactory
    # calculate depth from disparity. We use small-angle approximation for z due to diffraction

    
    x1 = X*f/Z
    x2 = x1
    y1 = Y*f/Z-Z/(2*B*f*kappa)
    y2 = Y*f/Z-Z/(2*B*f*kappa)
    
    y1 = pxNew*paramDict['imHeight']/2.-y1
    y2 = pxNew*paramDict['imHeight']/2.-y2
    x1 = pxNew*paramDict['imWidth']/2.-x1
    x2  = pxNew*paramDict['imWidth']/2.-x2
    
    return x1,y1, x2, y2 
    
    
    
def findObjects(trackIm, paramDict):
    """find objects after image segmentation etc and return pertinent parameters.""" 
    # invert image and get rid of some smaller noisy parts by gaussian smoothing
    trackIm = np.abs(trackIm)
    trackIm = gaussian(trackIm, sigma = 5)
    
    # apply threshold
    thresh = threshold_yen(trackIm)
    bw = opening(trackIm > thresh, square(5))
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
            locs.append([yob, xob])
    return np.array(locs)

def subtractBg(tmpIm, meanBrightness, bgs):
    """subtract either night or day background from image depending on brightness."""
    if np.mean(tmpIm) >= meanBrightness:
            return tmpIm - bgs[0]
    else:
            return tmpIm - bgs[1]

def initializePos(trackIm1, trackIm2, paramDict):
    # get starting locations for all stars in the image
    colors = ['w', 'r', 'b']
    currLocation = np.zeros((paramDict['NStars'],3))
    plt.figure('Image1: Click on each star in the image. We expect {} click(s) in each image.'.format(paramDict['NStars']), figsize=(12,8))
    plt.subplot(111)
    plt.imshow(trackIm1)
    locs1 = plt.ginput(paramDict['NStars'], timeout=0)
    plt.close()
    plt.figure('Image2: Click on each star in the image. We expect {} click(s) in each image.'.format(paramDict['NStars']), figsize=(12,8))
    plt.subplot(111)
    plt.imshow(trackIm2)
    for n in range(paramDict['NStars']):
        plt.plot(locs1[n][0], locs1[n][1],'o',color = colors[n], label = 'Last location Star {}'.format(n))
    #plt.plot(locs1[1][1], locs1[1][0],'o',color = 'w', label = 'Last location Star {}'.format(2))
    plt.legend(loc=0)
    locs2 = plt.ginput(paramDict['NStars'], timeout=0)
    plt.close()

    for n in range(paramDict['NStars']):
        currLocation[n] = calculate3DPoint(locs2[n][0], locs2[n][1], locs1[n][0], locs1[n][1], paramDict)
    return currLocation

def detectStars3D(impathSS1,impathSS2, bgSS1, bgSS2, paramDict, meanBrightess, show_figs= False, rgb = False):
    """traditional threshold-segment routine to detect specific sized objects after background subtraction."""
    imagesSS1 = loadSubset(impathSS1 , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['step'])
    imagesSS2 = loadSubset(impathSS2 , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['step'])
    nFrames = len(np.arange(paramDict['start'], paramDict['end'], paramDict['step']))
    
    coordinates = np.zeros((nFrames,paramDict['NStars'], 3))
    for imIndex in range(nFrames):
        # spit out some info about progress
        if imIndex%10==0:
            print 'step', imIndex
        
        # read image and subtract day/night Background
        trackIm1 = subtractBg(imread_convert(imagesSS1[imIndex], flag='SS1', rgb=rgb), meanBrightness, bgSS1)
        trackIm2 = subtractBg(imread_convert(imagesSS2[imIndex], flag='SS2', rgb=rgb), meanBrightness, bgSS2)
        # initialize stars
        if imIndex == 0:
            currLoc = initializePos(trackIm1, trackIm2, paramDict)
            coordinates[imIndex] = currLoc
            
        # detection, segmentation and labelling all packed into a nead function
            # also removes small objects
        objects1 = findObjects(trackIm1, paramDict)
        objects2 = findObjects(trackIm2, paramDict)
        #  calculate all possible x,y,z locations of possible objects
        locs3D = []
        locs2D = []
        for (y1,x1) in objects1:
            for (y2,x2) in objects2:
                # throw out large x displacements
                if (x2-x1) < paramDict['xTolerance']:
                    locs3D.append(calculate3DPoint(x1, y1, x2, y2, paramDict))
                    locs2D.append([x1,y1])
        locs3D = np.array(locs3D)
        # three cases: fewer objects, equal objects or more objects than stars
        if len(locs3D) == 0:
            currLoc = initializePos(trackIm1, trackIm2, paramDict)
            coordinates[imIndex] = currLoc
        
        # find object matches between 3D points and last known locations
        plt.figure()
        dist = np.zeros((paramDict['NStars'], len(locs3D)))
        for n in range(paramDict['NStars']):
            for lindex, loc in enumerate(locs3D):
                plt.plot(loc[0], loc[1], 'bo')
                
                dist[n, lindex] = np.sqrt(np.sum((currLoc[n] -loc)**2))
        print currLoc[n]
        plt.plot(currLoc[0], currLoc[1], 'ro')
        plt.show()
        # enter matches into coordinate base
        for n in range(paramDict['NStars']):
            loc, value = np.unravel_index(dist.argmin(), dist.shape), np.min(dist)
            print value
            if value < paramDict['minDistance']:
                # append to coordinate
                coordinates[imIndex,loc[0]] = locs3D[loc[1]]
                # delete coordinate from dstance map b setting to inf
                dist[loc[0]] = np.inf
                dist[:,loc[1]] = np.inf
                
            else:
                currLoc = initializePos(trackIm1, trackIm2, paramDict)
                coordinates[imIndex] = currLoc
        
    
    # show result
    plt.subplot(221)
    plt.imshow(bgSS1[0])
    #plt.plot(tracks[:,0,1])
    plt.plot(coordinates[:,:2])
    
    plt.show()



# this calculates a background based on day/night cycle and stores the images and assoc. data

impath =  "/media/monika/MyPassport/{}BothNs" # replace 'SS1' or 'SS2' with curly brackets {}
#impathSS2= "/media/monika/MyPassport/SS2NutellaSolo/"
analysisPath= "/media/monika/MyPassport/BothNs_Analysis/"



paramDict = {'start':0,
             'end':None,
             'step':1,
             'ext' : ".jpg",
            'daylengths':None,
            'framerate':3., #frames per minute
            'starsize':60,
            'NStars': 2, # How many stars should we look for,
            'minDistance':60, # in pixels per frame,
            'imHeight': 616,# actual image size (we rescale before saving)
            'imWidth': 820,
            'sensorPxX':3280,# maximum resolution of sensor
            'sensorPxY':2464,
            'lostFrames':10, # how many frames can a star not be visible until we reinitialize 
            'xTolerance': 25#how much jiggle in x axis is acceptable between cameras
}

bgCalc = False
tracking = True
# read the stored data and background images
if bgCalc:
    for flag in ['SS1', 'SS2']:
        calculateBackground(impath.format(flag), step=100, flag = flag, show_figs = True, rgb = True)
if tracking:
    
    paramDict['end'] = 200
    bgss1 = []
    bgss2 = []
    flag = 'SS1'
    bgSS1 = imread_convert(os.path.join(analysisPath, 'BG_Day_{}.jpg'.format(flag))).astype(np.float)\
    , imread_convert(os.path.join(analysisPath, 'BG_Night_{}.jpg'.format(flag))).astype(np.float)
    flag="SS2"
    bgSS2 = imread_convert(os.path.join(analysisPath, 'BG_Day_{}.jpg'.format(flag))).astype(np.float)\
    , imread_convert(os.path.join(analysisPath, 'BG_Night_{}.jpg'.format(flag))).astype(np.float)

    time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}.txt'.format(flag)), unpack=True)
    meanBrightness = np.mean(brightness)
    
    detectStars3D(impath.format('SS1'),impath.format('SS2'), bgSS1, bgSS2, paramDict, meanBrightness, show_figs= False, rgb = False)
    
        # detect objects based on thresholding and segmentation, filter by size and save coordinates.
        #d#etectStars(impath.format(flag), paramDict, meanBrightness, flag = flag, show_figs = False, rgb = True)


# use both trajectries to get 3d coordinates for the seastars
calculate_3d_coordinate(paramDict,analysisPath)

