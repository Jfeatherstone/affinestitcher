import numpy as np
import matplotlib.pyplot as plt

import numba

import cv2
from PIL import Image
import os
import tqdm

def filterBackground(img, tolerance=.05):
    """
    Filter out the background of the image by identifying the most
    prominent colours.

    Assumes that the background is significantly larger than the foreground,
    and therefore that the median colour of the image is the background.

    Parameters
    ----------

    img : numpy.ndarray[H,W]
        Image to be filtered

    tolerance : float
        The percent (as a decimal) difference from the median color for which a pixel
        will still be considered as background.

    Returns
    -------

    filteredImg : numpy.ndarray
        Image with background pixels set to (0,0,0) (black).
        
    """
    backgroundColor = np.mean([np.median(img[i,:], axis=0) for i in range(img.shape[0])], axis=0)
    editedImg = img

    editedImg[np.sqrt(np.sum((img - backgroundColor)**2, axis=-1)) < tolerance*np.max(img),:] = 0
    return editedImg


def equalizeSpatialGradients(img, strength=.5, debug=False):
    """

    """
    horizontalGrad = np.array([np.median(img[:,i,:], axis=0) for i in range(img.shape[1])])
    verticalGrad = np.array([np.median(img[i,:,:], axis=0) for i in range(img.shape[0])])

    horizontalCorr = horizontalGrad - np.mean(horizontalGrad, axis=0)
    verticalCorr = verticalGrad - np.mean(verticalGrad, axis=0)

    corr = np.transpose([np.add.outer(horizontalCorr[:,i].T, verticalCorr[:,i].T) for i in range(3)])

    corrImg = img - corr*strength
    corrImg = (corrImg - np.min(corrImg))
    corrImg /= np.max(corrImg)

    # Now make sure the image has the same type and min/max as the input
    if img.dtype == np.uint8:
        corrImg *= 255
        corrImg = corrImg.astype(np.uint8)

    if debug:
        correctedHorizontalGrad = np.array([np.median(corrImg[:,i,:], axis=0) for i in range(img.shape[1])])
        correctedVerticalGrad = np.array([np.median(corrImg[i,:,:], axis=0) for i in range(img.shape[0])])

        fig, ax = plt.subplots(1, 2)
        labels = ['Red', 'Green', 'Blue']
        for i in range(3):
            ax[0].plot(horizontalGrad[:,i], label=labels[i], c=labels[i])
            ax[0].plot(correctedHorizontalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[0].set_title('Horizontal')
        ax[0].legend()
        for i in range(3):
            ax[1].plot(verticalGrad[:,i], label=labels[i], c=labels[i])
            ax[1].plot(correctedVerticalGrad[:,i], label=f'Corr. {labels[i]}', c=f'tab:{labels[i]}')

        ax[1].set_title('Vertical')
        ax[1].legend()
        fig.tight_layout()
        plt.show()

    return corrImg

def checkImage(img):
    """
    Returns a numpy array of an img, regardless of whether given an image path
    or previously-loaded image.

    Useful for safely passing images to filtering/analysis methods, when the exact
    type isn't known a priori.

    Parameters
    ----------

    img : numpy.ndarray or PIL.Image or str
        Image or path to image to be returned as a numpy array.

    Returns
    -------

    npImg : numpy.ndarray
        Image as a numpy array.
    """
    if isinstance(img, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(cv2.imread(img), dtype=np.uint8)
    else:
        newFrame = np.array(img)

    return newFrame


def cropToContent(img, returnCorner=True):
    """
    Crop an image to the smallest possible size removing any information.

    Parameters
    ----------

    img : numpy.ndarray
        Image to be cropped.

    returnCorner : bool
        Whether to return the top left corner of the cropped image with
        respect to the original (True) or not (False). See Returns.

    Returns
    -------

    img OR (img, corner)

    img : numpy.ndarray
        Image cropped to content.

    corner : tuple
        Top left corner of the cropped image with respect to the original image.
        See `returnCorner`.
    """
    gray = np.mean(img, axis=-1)
    verticalBins = np.where(np.sum(gray, axis=0) > 0)
    if len(verticalBins[0]) > 0:
        leftBound, rightBound = verticalBins[0][0], verticalBins[0][-1]
    else:
        raise img

    horizontalBins = np.where(np.sum(gray, axis=1) > 0)
    if len(horizontalBins[0]) > 0:
        topBound, bottomBound = horizontalBins[0][0], horizontalBins[0][-1]
    else:
        raise img

    croppedImg = np.array(img)[topBound:bottomBound, leftBound:rightBound]
    return (croppedImg, (leftBound, topBound)) if returnCorner else croppedImg


def detectFeatures(images, nFeaturesMultiplier=1, bar=True, **kwargs):
    """
    Detect features in a set of images using the SIFT (Scale Invariant Feature Transform) method.

    Number of features to detect will be calculated for each image based on the size of the image
    and the value of `nFeaturesMultiplier`.

    Parameters
    ----------

    images : iterable of numpy.ndarray
       Images to detect features on. 

   
    nFeaturesMultiplier : float
        Multiplier for the number of features to identify for each image. Higher
        value is not guaranteed to generate more features, but will do so if possible.

    bar : bool
        Whether to show a progress bar (True) or not (False).

    kwargs : dict
        kwargs to be passed to cv2.SIFT_create().detectAndCompute(**kwargs).

    Returns
    -------

    features : iterable of dict
        List of dictionaries with keys ["keypoints", "descriptors"] describing the
        keypoints positions and feature descriptions respectively for each image.
    """

    # Load images
    imgArr = [checkImage(img) for img in images]

    features = [] 

    for i in tqdm.tqdm(range(len(imgArr)), desc='Feature detection') if bar else range(len(imgArr)):
        # 10 is arbitrary, just a reasonably selected value
        detector = cv2.SIFT_create(nfeatures=int(nFeaturesMultiplier*imgArr[i].shape[0]*imgArr[i].shape[1]/10))

        kp, des = detector.detectAndCompute(np.mean(imgArr[i], axis=-1).astype(np.uint8), None, **kwargs)
        features.append({"keypoints": kp,
                         "descriptors": des})

    return features


def computeAlignment(features1, features2, matcherType='flann', ratioThreshold=.5, slopeMatchingBinSize=1e-3, returnGoodMatches=False):
    """

    """
    if matcherType == 'flann':
        # FLANN
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    else:
        # Brute force
        matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(features1["descriptors"], features2["descriptors"], k=2)

    # Apply ratio test
    # No idea what this is, but a lower threshold is more selective
    good = []
    for m,n in matches:
        if m.distance < ratioThreshold*n.distance:
            good.append([m])

    da = []
    dr = []
    slope = []

    # Go through each match and compute what transform would be required to make the
    # features line up in the two images.
    for match in good:
        keypoint1 = features1["keypoints"][match[0].queryIdx]
        keypoint2 = features2["keypoints"][match[0].trainIdx]
        
        slope.append((keypoint1.pt[1] - keypoint2.pt[1]) / (keypoint1.pt[0] - keypoint2.pt[0] + 1e-8))

        dr.append(np.array(keypoint1.pt) - np.array(keypoint2.pt))
        da.append(keypoint1.angle - keypoint2.angle)

    if len(slope) > 0:
        # Instead of just taking the median of dr, it's more consistent to
        # calculate the slope of each dr (done above) and then take the dr
        # corresponding to the most prevalent slope (since these should just
        # be translations)

        # I am not sure if this method is applicable if we expect significant rotations between
        # the images, but for my uses I don't have to worry about that, so :|

        binSizeArr = np.logspace(-2, -5, 50)

        prominentDrArr = np.zeros((len(binSizeArr), 2))
        prominentDaArr = np.zeros(len(binSizeArr))
        countArr = np.zeros(len(binSizeArr))
        for i in range(len(binSizeArr)):
            bins = np.linspace(np.min(slope), np.max(slope), int((np.max(slope) - np.min(slope))/binSizeArr[i]))
            res = np.digitize(slope, bins)

            indices, counts = np.unique(res, return_counts=True)
            prominentDrArr[i] = np.mean(np.array(dr)[np.where(res == indices[np.argmax(counts)])], axis=0)
            prominentDaArr[i] = np.mean(np.array(da)[np.where(res == indices[np.argmax(counts)])], axis=0)
            countArr[i] = np.max(counts)

        # Split slopes into bins, so we can account for a slight bit of error (default is .1)
        #bins = np.linspace(np.min(slope), np.max(slope), int((np.max(slope) - np.min(slope))/slopeMatchingBinSize))
        #binnedSlope = np.digitize(slope, bins)
        # Check which slope appears the most
        #indices, counts = np.unique(binnedSlope, return_counts=True)

        # Average the actual values of dr and da for all of the entries whose
        # slopes were in the most common bin
        #avgDr = np.mean(np.array(dr)[np.where(binnedSlope == indices[np.argmax(counts)])], axis=0)
        #avgA = np.mean(np.array(da)[np.where(binnedSlope == indices[np.argmax(counts)])])
        avgDr = np.median(prominentDrArr, axis=0)
        avgA = np.median(prominentDaArr)

        if np.median(countArr) > 0:
            return (avgDr, avgA, np.median(countArr), good) if returnGoodMatches else (avgDr, avgA, np.median(countArr))
    
    return (np.array([np.nan, np.nan]), np.nan, 0, []) if returnGoodMatches else (np.array([np.nan, np.nan]), np.nan, 0)


BLENDER_OPTIONS = ['multiband',
                   'feather',
                   'no']

SEAM_FINDER_OPTIONS = ['color',
                       'colorgrad']

def stitchImages(imgArr, stitchCorners, stitchAngles=None, extraPadding=300, seamFinderType=SEAM_FINDER_OPTIONS[0], blenderType=BLENDER_OPTIONS[0], blenderStrength=1, crop=True):

    # Calculate the approximate corners of the stitched image
    left, right = np.min(stitchCorners[:,1]), np.max(stitchCorners[:,1])
    top, bottom = np.min(stitchCorners[:,0]), np.max(stitchCorners[:,0])

    # Add on some padding, which we can then crop off later
    approxStitchSize = (int(np.abs(right - left) + imgArr[0].shape[1] + extraPadding),
                        int(np.abs(bottom - top) + imgArr[-1].shape[0] + extraPadding))

    stitchedImage = Image.new('RGB', approxStitchSize)

    # For the seam finder and stitcher, we need the top left corners of
    # the images, as a list of tuples, ie.
    # [(x1,y1), (x2,y2), etc.]
    corners = [tuple(arr) for arr in stitchCorners.astype(np.int16)]
    imgSizes = [tuple(img.shape[:2])[::-1] for img in imgArr]

    seamFinder = cv2.detail_DpSeamFinder(seamFinderType.upper())
    # We proved a mask of just the full image, since we want to include as much as possible
    seamMasks = seamFinder.find(list(imgArr), corners, [255*np.ones(img.shape[:2], dtype=np.uint8) for img in imgArr])

    # Not sure what this thing is (region of interest?)
    dst_sz = cv2.detail.resultRoi(corners=corners, sizes=imgSizes)
    blendWidth = np.sqrt(dst_sz[2] * dst_sz[3]) * blenderStrength / 100

    if blenderType  == "no" or blendWidth < 1:
        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)

    elif blenderType == "multiband":
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(int((np.log(blendWidth) / np.log(2.0) - 1.0)))

    elif blenderType  == "feather":
        blender = cv2.detail_FeatherBlender()
        blender.setSharpness(1.0 / blendWidth)

    blender.prepare(dst_sz)

    # Feed each of the images into the blender, with the masks describing where the
    # seems will be (so they can be smoothed out)
    for i in range(len(imgArr)):
        blender.feed(cv2.UMat(np.array(imgArr[i]).astype(np.int16)), seamMasks[i], corners[i])
        
    stitchedImage = None
    stitchedMask = None
    stitchedImage, stitchedMask= blender.blend(stitchedImage, stitchedMask)

    stitchedImage, cropCorner = cropToContent(stitchedImage, True)

    return stitchedImage

