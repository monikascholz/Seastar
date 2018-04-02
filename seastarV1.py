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
import os
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.color import label2rgb
import matplotlib.patches as mpatches


impath= "/media/monika/MyPassport/SS1NutellaSolo/"
analysisPath= "/media/monika/MyPassport/SS1NutellaSolo_Analysis/"



paramDict = {'start':0,
             'end':None,
             'step':1,
             'ext' : ".jpg",
            'daylengths':None,
            'framerate':3., #frames per minute
            'starsize':400,
            'NStars': 1, # How many stars should we look for
}

def loadSubset(impath, extension, start = 0, end = None, step = 1):
    """load images with file extension extension following the pattern given by tuple.
    example: loadSubset('.', 'png', start=0, end=100, step =4) loads every 4th image in the directory up to the hundredth image."""
    imFileList = [os.path.join(impath,f) for f in os.listdir(impath) if f.endswith(extension)][start:end:step]
    return imFileList

def imread_convert(f):
    return img_as_float(io.imread(f, as_grey=True))
    #return rgb2gray()

def detectStars(impath, paramDict, meanBrightess):
    """traditional threshold-segment routine to detect specific sized objects after background subtraction."""
    allImFiles = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = paramDict['step'])
   
    objects = []
    for imIndex, imFile in enumerate(allImFiles):
        tmpIm = imread_convert(imFile)
         
        # subtract day/night Background
        if np.mean(tmpIm) >= meanBrightness:
            trackIm = tmpIm - dayBgIm
        else:
            trackIm = tmpIm - nightBgIm
        
        plt.imshow(trackIm)
        plt.show()
        # invert image and get rid of some smaller noisy parts by gaussian smoothing
        trackIm = 1-trackIm
        trackIm = gaussian(trackIm, sigma = 3)
        
        # apply threshold
        thresh = threshold_yen(trackIm)
        bw = opening(trackIm > thresh, square(5))
        # remove artifacts connected to image border
        cleared = clear_border(bw)
        # label image regions
        label_image = label(cleared)
    
        for rIndex, region in enumerate(regionprops(label_image, trackIm)):
            
        # take regions with large enough areas
            if region.area >= paramDict['starsize']:
                y0, x0 = region.centroid
                objects.append([imIndex, y0, x0, region.area, rIndex])
                # draw rectangle around segmented coins
    #            minr, minc, maxr, maxc = region.bbox
    #            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                      fill=False, edgecolor='red', linewidth=2)
    #            ax.add_patch(rect)
    objects = np.array(objects)
    np.savez(os.path.join(analysisPath, 'Tracks.npz'), tracks = objects)


def calculateBackground(impath, step, show_figs = True):
    """Calculate background images by median. First look at overall pixel change to get daylight/nightlight, 
    then use this info to calculate day and night backgrounds."""
    smallSet = loadSubset(impath , extension=paramDict['ext'], start = paramDict['start'],end = paramDict['end'], step = step)
    imgs = io.ImageCollection(smallSet, load_func = imread_convert)
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
    nactivity = np.mean(np.abs(np.diff(corrImgs, axis=0)),axis =(1,2))
    
    if show_figs:
        # calculate rough estimates
        plt.figure('Simple median background')
        plt.subplot(221)
        io.imshow(bgIm)
        plt.subplot(222)
        io.imshow(np.abs(imgs[0]-bgIm), show_cbar=True)
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
        plt.imshow(imgs[0]-dayBgIm)
        plt.subplot(224)
        plt.title('Subtracted night')
        plt.imshow(imgs[200]-nightBgIm)
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
        plt.axvline(meanBrightness)
        plt.xlabel('Change in Brightness')
        plt.ylabel('Acitvity')
        
        plt.show()
    
    io.imsave(os.path.join(analysisPath, 'BG_Day.jpg'), dayBgIm)
    io.imsave(os.path.join(analysisPath, 'BG_Night.jpg'), nightBgIm)
    
    np.savetxt(os.path.join(analysisPath, 'BgData.txt'), np.vstack([time[:-1], nactivity, brightness[:-1]]).T, header = "time, activity, brightness")
    
# this calculates a background based on day/night cycle and stores the images and assoc. data

#calculateBackground(impath, step=30, show_figs = True)#False)

# read the stored data and background images
dayBgIm, nightBgIm = imread_convert(os.path.join(analysisPath, 'BG_Day.jpg')).astype(np.float)\
, imread_convert(os.path.join(analysisPath, 'BG_Night.jpg')).astype(np.float)

time, nactivity, brightness = np.loadtxt(os.path.join(analysisPath, 'BgData.txt'), unpack=True)
meanBrightness = np.mean(brightness)


# detect objects based on thresholding and segmentation, filter by size and save coordinates.
#detectStars(impath, paramDict, meanBrightness)




plt.show()


# linking trajectories
