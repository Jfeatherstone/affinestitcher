import numpy as np
import matplotlib.pyplot as plt

import cv2
import tqdm

from scipy.fft import fft2

def getVideoFrames(videoPath):
    cam = cv2.VideoCapture(videoPath)

    frames = []

    while(True):
        ret, frame = cam.read()

        if ret:
            frames.append(frame.astype(np.uint8))

        else:
            break

    return frames


def identifyStableImages(images, derThreshold=.1, fftQuantile=.80, minSeqLen=2, differenceTol=.05, debug=False, bar=True):
    """
    Given a list of images (as read from a video) identify which ones are not
    blurry, and therefore suitable to be stitched together.
    """
    blurArr = np.zeros(len(images))
    
    if type(images[0]) is np.str_ or type(images[0]) is str:
        imgArr = [cv2.imread(p) for p in images]
    else:
        imgArr = images
        
    for i in tqdm.tqdm(range(len(blurArr)), desc='Identifying stable images'):# if bar else range(len(blurArr)):
        # Grayscale
        img = np.mean(imgArr[i], axis=-1)
        # FFT
        fftResult = fft2(img).flatten()
        # Take average of top {fftQuantile} of frequencies
        blurArr[i] = np.abs(np.mean(np.sort(fftResult)[int(len(fftResult)*fftQuantile):]))
        
    # Take derivative of the blur, and throw out points that have a large value
    derBlur = blurArr[1:] - blurArr[:-1]
    derBlur = np.where(np.abs(derBlur) > derThreshold*np.mean(np.abs(derBlur)), np.nan, derBlur) 

    # Locate where the proper values are, so we can get the center of each group
    splits = np.where(1 - np.isnan(derBlur))[0]

    # The groups of indices for each consecutive non-nan chain
    indexGroups = np.split(splits, np.where(np.diff(splits) != 1)[0] + 1)

    # Calculate how different each frame is from the others in this chain,
    # since we may want to include more than one

    stableImageIndices = []

    for i in range(len(indexGroups)):
        diffMat = np.array([[np.sqrt(np.sum((imgArr[j] - imgArr[k])**2)) / (imgArr[j].shape[0]*imgArr[j].shape[1]) for j in indexGroups[i]] for k in indexGroups[i]])

        repImages = []
        for j in range(len(indexGroups[i])):
            similarImages = np.where(diffMat[j] < differenceTol)[0]
            if len(similarImages) > 0:
                repImages.append(indexGroups[i][np.argmax(blurArr[indexGroups[i][similarImages]])])
            else:
                repImages.append(indexGroups[i][j])

        stableImageIndices += list(np.unique(repImages))

    stableImageIndices = np.array(stableImageIndices)

    # Alternative, simpler approach
    # Take the best frame of each conecutive non-nan chain
    #stableImageIndices = [np.max(s) for s in indexGroups if not np.isnan(s[0]) and len(s) >= minSeqLen]

    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(max(len(stableImageIndices)/10, 5), 4))
        
        ax.plot(blurArr)
        ax.set_xlabel('Image index')
        ax.set_ylabel(r'$\langle f_{FFT} \rangle$')
        
        for i in range(len(stableImageIndices)):
            ax.axvline(stableImageIndices[i], linestyle='--', alpha=.2, c='r')
        
        plt.show()
        
    return stableImageIndices

