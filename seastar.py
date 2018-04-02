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
    
    
def calculate_3d_coordinate(paramDict,analysisPath):
    tracksSS1 = np.load(os.path.join(analysisPath, 'Tracks_{}.npz'.format('SS1')))['tracks']
    tracksSS2 = np.load(os.path.join(analysisPath, 'Tracks_{}.npz'.format('SS2')))['tracks']
    
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
    
    y1R, x1R = np.squeeze(pxNew*tracksSS1[:,:,1]), np.squeeze(pxNew*tracksSS1[:,:,2])
    y2R, x2R = np.squeeze(pxNew*tracksSS2[:,:,1]), np.squeeze(pxNew*tracksSS2[:,:,2])
    
    y1R = pxNew*paramDict['imHeight']/2.-y1R
    y2R = pxNew*paramDict['imHeight']/2.-y2R
    x1R = pxNew*paramDict['imWidth']/2.-x1R
    x2R = pxNew*paramDict['imWidth']/2.-x2R
    # smooth out large jumps
    wl = 13
    
#    x1 = savgol_filter(x1R,wl,3)
#    y1 = savgol_filter(y1R,wl,3)
#    x2 = savgol_filter(x2R,wl,3)
#    y2 = savgol_filter(y2R,wl,3)
    x1 = medfilt(x1R,wl)
    y1 = medfilt(y1R,wl)
    x2 = medfilt(x2R,wl)
    y2 = medfilt(y2R,wl)
    
    # correct for unintended x-shift
    x1 -= x1[0]-x2[0]
    # calculate depth from disparity. We use small-angle approximation for z due to diffraction
    Z = -B*f/(y1-y2)*kappa
    X = (x1+x2)/2.*Z/f
    Y = (y1+y2)/2.*Z/f
    
    # show output
    plt.subplot(221)
    plt.plot(x1R,y1R, alpha=0.5)
    plt.plot(x2,y2)
    plt.plot(x1,y1)
    plt.plot(x2R,y2R, alpha=0.5)
    plt.plot(x1[0], y1[0], 'ro')
    plt.ylabel('y (a.u.)')
    plt.xlabel('x (a.u.)')
    plt.subplot(222)
    plt.plot(X,Y)
    plt.ylabel('y (pcm)')
    plt.xlabel('x (cm)')
    plt.subplot(223)
    plt.plot(Z,Y, 'o')
    plt.ylabel('y (cm)')
    plt.xlabel('z (cm)')
    plt.subplot(224)
    plt.plot(X,Z)
    plt.ylabel('z (cm)')
    plt.xlabel('x (cm)')
    plt.tight_layout()
    plt.show()
    
    
    # tracking error
    plt.hist(Y-np.mean(Y), 30, normed=True)
    plt.ylabel('Normalized counts')
    plt.xlabel('tracking error (cm)')
    #ax = plt.subplot(212, projection='3d')
    #ax.plot(x1,y1,z1)
    plt.tight_layout()
    plt.show()
    
    

def detectStars(impath, paramDict, meanBrightess, flag, show_figs= False, rgb = False):
    """traditional threshold-segment routine to detect specific sized objects after background subtraction."""
    allImFiles = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['step'])
     
    lostFrames = 0
    color = ['w', 'r', 'b']
    tracks = np.zeros((len(allImFiles),paramDict['NStars'], 5))
    for imIndex, imFile in enumerate(allImFiles):
        if imIndex%100==0:
            print 'step', imIndex
        tmpIm = imread_convert(imFile, flag, rgb)
        # subtract day/night Background
        if np.mean(tmpIm) >= meanBrightness:
            trackIm = tmpIm - dayBgIm
        else:
            trackIm = tmpIm - nightBgIm
    
        if imIndex == 0:
        # get starting locations for all stars in the image
            plt.figure('Click on each star. We expect {} click(s).'.format(paramDict['NStars']), figsize=(12,8))
           
            plt.imshow(trackIm)
            locs = plt.ginput(paramDict['NStars'], timeout=0)
            plt.close()
            for n in range(paramDict['NStars']):
                tracks[imIndex, n] = [imIndex, locs[n][1], locs[n][0], 0, 1]
            continue
        
        # if we lost the stars for too long, reinitialize
        if lostFrames > paramDict['lostFrames']:
            plt.figure('Click on each star. We expect {} click(s). Click in the order indicated by the legend'.format(paramDict['NStars']), figsize=(12,8))
            plt.subplot(211)
            plt.imshow(trackIm)
            plt.subplot(212)
            plt.imshow(tmpIm)
            plt.tight_layout()
            for n in range(paramDict['NStars']):
                plt.plot(tracks[imIndex-1, n, 2], tracks[imIndex-1, n, 1],'o',color = color[n], label = 'Last location Star {}'.format(n))
                plt.plot(tracks[:imIndex-1,n,2], tracks[:imIndex-1,n,1], '-', color = color[n])
            plt.legend()
            locs = plt.ginput(paramDict['NStars'], timeout=0)
            plt.close()
            for n in range(paramDict['NStars']):
                tracks[imIndex, n] = [imIndex, locs[n][1], locs[n][0], 0, 1]
            lostFrames = 0
            continue
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
        objects = regionprops(label_image, trackIm)
        # if we have no points in the image, append the previous points
        if show_figs:
            plt.subplot(211)
            plt.imshow(trackIm)
            for n in range(paramDict['NStars']):
                plt.plot(tracks[imIndex-1, n, 2], tracks[imIndex-1, n, 1],'o',color = color[n], label = 'Last location Star {}'.format(n))
                plt.plot(tracks[:imIndex-1,n,2], tracks[:imIndex-1,n,1], '-', color = color[n], alpha=0.5)
            plt.subplot(212)
            plt.imshow(label_image)
            plt.show()
        tmpPoints = []
        if len(objects) == 0:
            print imIndex,'no objects'
            for n in range(paramDict['NStars']):
                tracks[imIndex, n] = tracks[imIndex-1, n]
                # set 0 flag for repeat entry
                tracks[imIndex, n, -1] = 0
                # update timestamp
                tracks[imIndex, n][0] += 1 
                lostFrames +=1
            continue
            
        dist = np.zeros((paramDict['NStars'], np.max([len(objects),paramDict['NStars']])))
        tmpPoints = np.zeros((np.max([len(objects),paramDict['NStars']]), 5))
        for n in range(paramDict['NStars']):
            
            _, y0, x0, _, _ = tracks[imIndex-1][n]
            for rIndex, region in enumerate(objects):
            # take regions with large enough areas
                
                yob, xob = region.centroid
                 
                #objects.append([imIndex, y0, x0, region.area, rIndex])
                # calculate distance
                dist[n,rIndex] = np.sqrt((xob-x0)**2+(yob-y0)**2)
                
                tmpPoints[rIndex] = [imIndex, yob, xob, region.area, 1]
        
        # match points to trajectories
                        # while we have positive entries
    
        accepted = 0
        while accepted < paramDict['NStars']:
            
            loc, value = np.unravel_index(dist.argmin(), dist.shape), np.min(dist)
            starIndex = loc[0]
            #print 'acc:', accepted, value, loc, starIndex
            #print loc, value, tmpPoints
            if value < paramDict['minDistance'] and tmpPoints[loc[1]][-2]>paramDict['starsize']:
                #print 'found you!', tmpPoints[loc[1]][-2]
                tracks[imIndex, starIndex] = tmpPoints[loc[1]]
                accepted +=1
            else:
                # if the smallest distance is already too large, repeat last entry and set repeat flag
                tracks[imIndex, starIndex] = tracks[imIndex-1,starIndex]
                # set 0 flag for repeat entry
                tracks[imIndex, starIndex][-1] = 0
                # update timestamp
                tracks[imIndex, starIndex][0] += 1 
                lostFrames += 1
                accepted += 1
            # set distances to infinity for the track that already has a match
            dist[starIndex] = np.inf
            dist[:,loc[1]] = np.inf
            
    
             
    tracks = np.array(tracks)
    np.savez(os.path.join(analysisPath, 'Tracks_{}.npz'.format(flag)), tracks = tracks)
    # show result
    plt.subplot(221)
    plt.imshow(dayBgIm)
    #plt.plot(tracks[:,0,1])
    for n in range(paramDict['NStars']):
        plt.plot(tracks[:,n,2], tracks[:,n,1], '-', color = color[n])
    #plt.plot(tracks[:,1,1], tracks[:,1,2])
    plt.subplot(223)
    plt.plot(tracks[:,0,2], tracks[:,0,1], '-', color = 'r')
    
    plt.subplot(224)
    plt.plot(tracks[:,1,2], tracks[:,1,1], '-', color = 'b')
    plt.show()


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
            'starsize':50,
            'NStars': 2, # How many stars should we look for,
            'minDistance':30, # in pixels per frame,
            'imHeight': 616,# actual image size (we rescale before saving)
            'imWidth': 820,
            'sensorPxX':3280,# maximum resolution of sensor
            'sensorPxY':2464,
            'lostFrames':10 # how many frames can a star not be visible until we reinitialize 
}

bgCalc = False
tracking = True
# read the stored data and background images
if bgCalc:
    for flag in ['SS1', 'SS2']:
        calculateBackground(impath.format(flag), step=100, flag = flag, show_figs = True, rgb = True)
if tracking:
    #paramDict['end'] = 200
    for flag in ['SS1', 'SS2']:
        dayBgIm, nightBgIm = imread_convert(os.path.join(analysisPath, 'BG_Day_{}.jpg'.format(flag))).astype(np.float)\
        , imread_convert(os.path.join(analysisPath, 'BG_Night_{}.jpg'.format(flag))).astype(np.float)
        
        time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData_{}.txt'.format(flag)), unpack=True)
        meanBrightness = np.mean(brightness)
        
        
        # detect objects based on thresholding and segmentation, filter by size and save coordinates.
        detectStars(impath.format(flag), paramDict, meanBrightness, flag = flag, show_figs = False, rgb = True)


# use both trajectries to get 3d coordinates for the seastars
calculate_3d_coordinate(paramDict,analysisPath)

