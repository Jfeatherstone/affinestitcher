import numpy as np
from PIL import Image

import affinestitcher as af
import os
import argparse

import matplotlib.pyplot as plt 
import tqdm

import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument(dest='inputPath', type=str, help="Path to directory containing image files or video")

    parser.add_argument('-m', type=str, default='flann', dest='matcherType', help='Matcher type')
    parser.add_argument('-s', type=str, default='color', dest='seamFinderType', help='Seam finder type')
    parser.add_argument('-b', type=str, default='multiband', dest='blenderType', help='Blender type')
    parser.add_argument('-o', type=str, default='stitched_image.png', dest='outputFile', help='Output file name')
    parser.add_argument('-d', type=str, default='', dest='debugDir', help='Debug directory')

    parser.add_argument('--dry', dest='dry', action='store_const', const=True, default=False)

    args = parser.parse_args()

    debug = False
    if len(args.debugDir) > 0:
        debug = True
        if not os.path.exists(args.debugDir):
            os.mkdir(args.debugDir)

    imageExtension = 'png'
    videoExtension = 'avi'
    fileTemplateName = 'image'

    if args.inputPath.split('.')[-1] == videoExtension:
        print('Reading video frames...')
        images = af.getVideoFrames(args.inputPath)#[:250]

    elif os.path.isdir(args.inputPath):
        images = [args.inputPath + '/' + i for i in os.listdir(args.inputPath) if imageExtension in i]
        imageIndexing = [int(img.split(fileTemplateName)[-1].split('.')[0]) for img in images]

        images = np.array(images)[np.argsort(imageIndexing)]
    else:
        raise Exception('Invalid input file(s) given')

    stableImageIndices = af.identifyStableImages(images, derThreshold=.05, debug=False)
    #print(f'Found {len(stableImageIndices)} stable images')

    imgArr = [af.checkImage(images[i]) for i in tqdm.tqdm(stableImageIndices, desc='Loading stable images')]

    features = af.detectFeatures(imgArr)

    # These will have shape (p,i,2) and (p,i), where p is the patch
    # index, and i is the image index. In the case that there is some
    # disconnect between any two frames, this means every subsequent
    # point won't fail
    stitchOffsetArr = [[[0,0]]]
    stitchAngleArr = [[0]]
    confidenceArr = []
    featureCheckKernel = 2
    auxillaryImageCheckKernel = 15
    skipIndices = []

    for i in tqdm.tqdm(range(len(imgArr)-1), desc='Computing alignment'):
        if i in skipIndices:
            continue

        dr, da, conf = af.computeAlignment(features[i], features[i+1], args.matcherType)
        
        if conf > 0:
            # If we find a match, add it to the most recent patch
            stitchOffsetArr[-1].append(dr)
            stitchAngleArr[-1].append(da)
            confidenceArr.append(conf)
        else:
            # If we don't find a match, see if the immediately following
            # frames can be matched up
            # DEBUG
            #print(f'Invoking kernel feature check for index {i}')
            for j in range(min(featureCheckKernel, len(imgArr)-i-2)):
                foundAlignment = False
                dr2, da2, conf = af.computeAlignment(features[i], features[i+2+j], args.matcherType)

                if conf > 0:
                    skipIndices += list(range(i+1,i+j+2))
                    stitchOffsetArr[-1].append(dr2)
                    stitchAngleArr[-1].append(da2)
                    confidenceArr.append(conf)
                    foundAlignment = True
                    # DEBUG
                    #print(f'Found match with index {i+2+j}')
                    break
            
            if not foundAlignment:
                # If we *still* don't find a match, we can try to use some of the
                # potentially unstable images 
                neighboringImages = [af.checkImage(images[min(max(0, stableImageIndices[i]+j), len(images))]) for j in range(1, min(auxillaryImageCheckKernel, stableImageIndices[i+1] - stableImageIndices[i]))]
                neighboringFeatures = af.detectFeatures(neighboringImages, bar=False)
                currentFrameMatches = []
                nextFrameMatches = []
                # DEBUG
                #print('Attempting to use blurry images to match')
                
                for j in range(len(neighboringFeatures)):
                    # We want the good matches back here so we can make a debug image
                    currDr, currDa, currConf, currGoodMatches = af.computeAlignment(features[i], neighboringFeatures[j], args.matcherType, returnGoodMatches=True)
                    nextDr, nextDa, nextConf, nextGoodMatches = af.computeAlignment(neighboringFeatures[j], features[i+1], args.matcherType, returnGoodMatches=True)

                    if currConf > 0 and nextConf > 0:
                        stitchOffsetArr[-1].append([currDr[0] + nextDr[0], currDr[1] + nextDr[1]])
                        stitchAngleArr[-1].append(currDa + nextDa)
                        confidenceArr.append((currConf + nextConf) / 2)
                        foundAlignment = True
                        # DEBUG
                        #print('Found alignment using blurry frame')

                        if debug:
                            fig, ax = plt.subplots(1, 3, figsize=(16,5))

                            drawParams = dict(matchColor = (0,255,0),
                                             singlePointColor = (255,0,0),
                                             flags = cv2.DrawMatchesFlags_DEFAULT)

                            firstMatchImg = cv2.drawMatchesKnn(imgArr[i], features[i]["keypoints"],
                                                     neighboringImages[j], neighboringFeatures[j]["keypoints"],
                                                     currGoodMatches, None, **drawParams)
                            secondMatchImg = cv2.drawMatchesKnn(neighboringImages[j], neighboringFeatures[j]["keypoints"],
                                                     imgArr[i+1], features[i+1]["keypoints"],
                                                     nextGoodMatches, None, **drawParams)

                            manualImage = Image.new('RGBA', (2000, 2000))

                            img0 = Image.fromarray(imgArr[i].astype(np.uint8))
                            #img0.putalpha(255)

                            img1 = Image.fromarray(imgArr[i+1].astype(np.uint8))
                            img1.putalpha(180)

                            manualImage.paste(img0, (500,500))
                            manualImage.paste(img1, (500 + int(stitchOffsetArr[-1][-1][0]), 500 + int(stitchOffsetArr[-1][-1][1])), img1.convert('RGBA'))

                            ax[0].imshow(firstMatchImg)
                            ax[1].imshow(secondMatchImg)
                            ax[2].imshow(manualImage)
                            fig.tight_layout()
                            plt.savefig(f'{args.debugDir}/blurry_match_{i}.png')
                            plt.close()

                        break

                # Finally, if we still haven't found anything, we need to start a new patch
                if not foundAlignment:
                    if debug:
                        Image.fromarray(imgArr[i]).save(f'{args.debugDir}/mismatch_{i}_0.png')
                        Image.fromarray(imgArr[i+1]).save(f'{args.debugDir}/mismatch_{i}_1.png')
                    # DEBUG
                    #print('Starting new patch')
                    stitchOffsetArr.append([[0,0]])
                    stitchAngleArr.append([0])
                    confidenceArr.append(0)

    # Cut out the images we removed
    for ind in skipIndices:
        del imgArr[ind]
        stableImageIndices = np.delete(stableImageIndices, np.where(stableImageIndices == ind))

    # DEBUG
    #plt.plot(confidenceArr)
    #plt.savefig('conf.png')
    #plt.close()

    patchIterCount = 0
    while len(stitchOffsetArr) > 1:
        # If we have more than one patch, we now need to stitch those
        # together

        # This is done by stitching each patch fully together, then
        # computing features again.
        patchImgStartIndices = np.cumsum([0] + [len(arr) for arr in stitchOffsetArr[:-1]])
        patchImgEndIndices = np.append(patchImgStartIndices[1:], len(imgArr))
        # DEBUG
        #print(f'start indices: {patchImgStartIndices}')
        #print(f'end indices: {patchImgEndIndices}')

        patchImgArr = []
    
        for p in tqdm.tqdm(range(len(stitchOffsetArr)), desc=f'Stitching patches (iteration {patchIterCount})'):
            patchStitchPositions = np.cumsum(stitchOffsetArr[p], axis=0)
            #print(len(imgArr[patchImgStartIndices[p]:patchImgEndIndices[p]]))

            stitchedPatch = af.stitchImages(imgArr[patchImgStartIndices[p]:patchImgEndIndices[p]], patchStitchPositions, seamFinderType=args.seamFinderType, blenderType=args.blenderType)

            patchImgArr.append(stitchedPatch)

        if debug:
            fig, ax = plt.subplots(1, len(patchImgArr))

            for i in range(len(ax)):
                ax[i].imshow(patchImgArr[i])
                Image.fromarray(patchImgArr[i].astype(np.uint8)).save(f'images/patch_{i}.png')

            plt.savefig(f'{args.debugDir}/patch_iter_{patchIterCount}.png')
            plt.close()

        patchFeatures = af.detectFeatures(patchImgArr, bar=False)

        patchStitchOffsetArr = [[[0,0]]]
        patchStitchAngleArr = [[0]]

        for i in range(len(patchImgArr)-1):
            dr, da, conf = af.computeAlignment(patchFeatures[i], patchFeatures[i+1], args.matcherType, ratioThreshold=.5)
            
            if conf > 0:
                # If we find a match, add it to the most recent patch
                patchStitchOffsetArr[-1].append(dr)
                patchStitchAngleArr[-1].append(da)
            else:
                # If we don't find a match, we need to start a new patch
                patchStitchOffsetArr.append([[0,0]])
                patchStitchAngleArr.append([0])

        # Make sure we've actually gotten better
        if len(patchStitchOffsetArr) >= len(stitchOffsetArr):
            raise Exception('Patching image yields no improvement!')

        newStitchOffsetArr = []

        patchLenArr = [0] + list(np.cumsum([len(arr) for arr in patchStitchOffsetArr]))
        includedOriginalPatches = [np.arange(patchLenArr[i-1], patchLenArr[i]) for i in range(1, len(patchLenArr))]

        # DEBUG
        #print(includedOriginalPatches)
        #print(patchStitchOffsetArr)

        # Now stitch together whatever patches we can, and potentially repeat the process
        for p in range(len(patchStitchOffsetArr)):
            newStitchOffsetArr.append([])

            for i in range(len(includedOriginalPatches[p])):
                originalOffsets = np.array(stitchOffsetArr[includedOriginalPatches[p][i]])
                newStitchOffsetArr[-1] += list([np.array(patchStitchOffsetArr[p][i])])
                newStitchOffsetArr[-1] += list(originalOffsets[1:])

            newStitchOffsetArr[-1] = np.array(newStitchOffsetArr[-1])

        stitchOffsetArr = newStitchOffsetArr
        patchIterCount += 1


    stitchOffsetArr = np.array(stitchOffsetArr[0])
    stitchAngleArr = np.array(stitchAngleArr[0])

    absoluteStitchPositions = np.cumsum(stitchOffsetArr, axis=0)
       
    if args.dry:
        # DEBUG
        #print(absoluteStitchPositions)
        #print(np.shape(absoluteStitchPositions))

        plt.scatter(absoluteStitchPositions[:,0], absoluteStitchPositions[:,1])
        plt.scatter(absoluteStitchPositions[-1,0], absoluteStitchPositions[-1,1], c='r')
        plt.scatter(absoluteStitchPositions[0,0], absoluteStitchPositions[0,1], c='g')
        plt.savefig(args.outputFile)
        plt.close()
    
    else:
        print('Stitching image...')
        stitchedImg = af.stitchImages(imgArr, absoluteStitchPositions, seamFinderType=args.seamFinderType, blenderType=args.blenderType)

        Image.fromarray(np.array(stitchedImg, dtype=np.uint8)).save(args.outputFile)

