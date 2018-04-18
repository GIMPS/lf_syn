import os
import os.path
import numpy as np
from PIL import Image
from math import floor
from InitParam import param,novelView,inputView,get_folder_content
from scipy.interpolate import interp2d
import torch




def make_dir(inputPath):
    if not os.path.exists(inputPath):
        os.makedirs(inputPath)

def load_image( path ) :
    img = Image.open( path )
    return  np.asarray( img.convert('RGB') )

def crop_img(input,pad):
    return input[pad:- pad, pad: - pad]

def get_patches(input, patchSize, stride):

    [height, width, depth] = input.shape

    numPatches = (floor((width - patchSize) / stride) + 1) * (floor((height - patchSize) / stride) + 1)
    patches = np.zeros((patchSize, patchSize, depth, numPatches))

    count = 0
    for iX in np.arange(0,width-patchSize + 1,stride):
        for iY in np.arange(0,height - patchSize + 1,stride):
            patches[:,:,:, count] = input[iY: iY + patchSize, iX: iX + patchSize,:]
            count = count + 1
    return patches

def get_num_patches():
    height=param.cropHeight
    width = param.cropWidth
    patchSize =param. patchSize
    stride = param.stride

    numPatchesX = floor((width - patchSize) / stride) + 1
    numPatchesY = floor((height - patchSize) / stride) + 1
    numPatches = numPatchesY * numPatchesX
    return numPatches

def get_img_pos(ind):
    pos = ind / (param.origAngRes - 1)
    return pos


def defocus_response(input):
    curMean = np.nanmean(input, 2)
    curMean[np.isnan(curMean)] = 0
    output = curMean
    return output

def corresp_response(input):
    inputVar = np.nanvar(input, 2,ddof=1)
    inputVar[np.isnan(inputVar)] = 0
    output = np.sqrt(inputVar)
    return output


def pad_with_one(input, finalLength):
    output = list(input)+ np.ones((1, finalLength - len(input)),dtype=np.uint8).flatten().tolist()
    return output

def save_hdf(fileName, datasetName, input, inDims, startLoc, chunkSize, createFlag, arraySize= 1):
    import warnings
    warnings.filterwarnings("ignore")
    import h5py

    f = h5py.File(fileName, "a")
    if createFlag:
        dset = f.create_dataset(datasetName, (*inDims[0:- 1], arraySize), dtype='f',chunks=True)
        print(dset.shape)
        print(dset.maxshape)
    else:
        dset=f.get(datasetName)

    sliceIdx=[]
    for i in range(len(inDims)):
        #_handle_simple in h5py/slection.py does not handle length=1 slices properly
        if inDims[i]== 1:
            idx=startLoc[i]
        else:
            idx=slice(startLoc[i],startLoc[i]+inDims[i])
        sliceIdx.append(idx)
    while input.shape[-1]== 1:
        input=input.flatten()
    dset.write_direct(input.astype('float32'),dest_sel=tuple(sliceIdx))
    startLoc[-1] = startLoc[-1] + inDims[-1]
    f.close()
    print("Done")
    return startLoc

def warp_images(disparity, input, delY, delX):
    [h, w, numImages] = disparity.shape
    [X, Y] = np.mgrid[0:w, 0: h]
    c = input.shape[2]

    output = np.zeros((h, w, c, numImages), 'flaot')
    for j in range(0, numImages):
        for i in range(0, c):
            curX = X + delX(j) * disparity[:,:, j]
            curY = Y + delY(j) * disparity[:,:, j]

            #output[:,:, i, j] = interp2(X, Y, input[:,:, i, j], curX, curY, 'cubic', nan)#todo
    return output


def warp_all_images(images,depth,refPos):
    [h,w,c,numImages]=images.shape
    numInputViews=len(inputView.Y)
    #######todo:gather
    warpedImages=np.zeros((h,w,c,numImages),'float')
    for i in range(0,numInputViews):
        deltaY=inputView.Y[i],-refPos[0,:]
        deltaX=inputView.X[i],-refPos[1,:]
        warpedImages[:, :, (i-1)*3+1 : i*3, :]= warp_images(depth, images[:, :, (i-1)*3+1:i*3, :], deltaY, deltaX)
    return warpedImages


def prepare_depth_features(inputLF, deltaY, deltaX):
    depthResolution = param.depthResolution
    deltaDisparity = param.deltaDisparity
    ##convert the input rgb light field to grayscale
    (height, width, _, angHeight, angWidth) = inputLF.shape
    grayLF = np.zeros((height, width, angHeight, angWidth))
    for i in range(0, angHeight):
        for j in range (0, angWidth):
            im = Image.fromarray(inputLF[:,:,:, i, j])
            grayLF[:,:, i, j] = np.asanyarray(im.convert('L'))

    defocusStack = np.zeros((height, width, depthResolution))
    correspStack = np.zeros((height, width, depthResolution))
    featuresStack = np.zeros((height, width, 200))
    delta = 2 * deltaDisparity / (depthResolution - 1)
    # print(delta)
    indDepth =0

    for curDepth in np.arange( - deltaDisparity, deltaDisparity+delta,delta):
        #if indDepth% 10 == 0:
            #print()
        #todo: progress bar
        shearedLF = np.zeros((height, width, angHeight * angWidth))
        X=np.arange(0,width)
        Y=np.arange(0,height)

        #backward warping all the input images using each depth level (see Eq. 5)

        indView = 0
        for iax in range(0, angWidth):
            for iay in range(0, angHeight):
                curY = Y + curDepth * deltaY[indView]
                curX = X + curDepth * deltaX[indView]
                ip=interp2d(X,Y,grayLF[:,:, iay, iax],kind='cubic')
                shearedLF[:,:, indView] =ip(curX, curY)
                indView = indView + 1
        #computing the final mean and variance features for depth level using Eq. 6

        defocusStack[:,:, indDepth] = defocus_response(shearedLF)
        correspStack[:,:, indDepth] = corresp_response(shearedLF)

        if (indDepth+1)%10 == 0:
            print((indDepth+1) / depthResolution * 100)

        indDepth = indDepth + 1

    featuresStack[:,:, 0: 100] = defocusStack.astype('float32')
    featuresStack[:,:, 100: 200] = correspStack.astype('float32')
    # print(grayLF)
    return featuresStack


def prepare_color_features(depth,images,refPos):
    images=crop_img(images,param.depthBorder)
    warpedImages=warp_all_images(images,depth,refPos)
    indNan=np.isnan(warpedImages)
    warpedImages[indNan]=0

    warpedImages=torch.from_numpy(warpedImages)
    if param.useGPU:
        warpedImages=warpedImages.cuda()
    [h,w,_]=depth.shape
    refPos = refPos.reshape((2, 1, 1,-1))
    colorFeatures = np.concatenate(( depth, warpedImages, np.tile(refPos[0,:,:,:]-1.5, (h, w, 1, 1)), np.tile(
        refPos[1,:,:,:]-1.5, (h, w, 1, 1))),axis=3)
    return colorFeatures, indNan


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def read_illum_images(scenePath):
    """Read from illum images.

    :return:
    """
    numImgsX=14
    numImgsY=14

    inputImg = load_image(scenePath)
    inputImg=im2double(inputImg)
    h = inputImg.shape[0] // numImgsY
    w = inputImg.shape[1] // numImgsX
    fullLF = np.zeros((h, w, 3, numImgsY, numImgsX),dtype=np.uint8)
    for ax in range(numImgsX):
        for ay in range(numImgsY):
            fullLF[:, :, :, ay, ax] = inputImg[ay::numImgsY, ax::numImgsX,:]
    if h == 375 and w == 540:
        fullLF = np.pad(fullLF, ((0, 1),(0,1),(0,0),(0,0),(0,0)), mode='constant',constant_values=0)
    if h == 375 and w == 541:
        fullLF = np.pad(fullLF, ((0,1),(0,0),(0,0),(0,0),(0,0)), mode='constant',constant_values=0)
    fullLF = fullLF[:, :, :, 3:11, 3:11]
    inputLF = fullLF[:, :, :, 1:8:6,1:8:6]
    print("curInputLF size is ")
    print(inputLF.shape)
    return fullLF,inputLF

def compute_training_examples(curFullLF, curInputLF):

    cropSize = param.cropSizeTraining
    numRefs = param.numRefs
    patchSize = param.patchSize
    stride = param.stride
    origAngRes = param.origAngRes

    #########preparing input images
    (height,width,_,_,_)=curInputLF.shape
    inImgs=curInputLF.reshape((height,width,-1))

    inImgs = crop_img(inImgs, cropSize)
    pInImgs = get_patches(inImgs, patchSize, stride)
    pInImgs=np.tile(pInImgs,(1,1,1,numRefs))


    ####selecting random references

    numSeq = np.random.permutation(origAngRes ** 2)
    refInds = numSeq[0:numRefs]

    ##########initializing the arrays
    numPatches = get_num_patches()
    pInFeat = np.zeros((patchSize, patchSize, param.numDepthFeatureChannels, numPatches * numRefs))
    pRef = np.zeros((patchSize, patchSize, 3, numPatches * numRefs))
    refPos = np.zeros((2, numPatches * numRefs))

    for ri in range(0,numRefs):

        print('Working on random reference %d of %d: '%( ri+1, numRefs))
        curRefPos = type('', (), {})()
        curRefInd = type('', (), {})()
        [curRefInd.Y, curRefInd.X] = np.unravel_index(refInds[ri],[origAngRes, origAngRes], 'F')
        curRefPos.Y = get_img_pos(curRefInd.Y)
        curRefPos.X = get_img_pos(curRefInd.X)

        wInds = np.arange((ri - 1) * numPatches, ri * numPatches)

        #preparing reference
        ref = curFullLF[:,:,:, curRefInd.Y, curRefInd.X]
        ref = crop_img(ref, cropSize)
        pRef[:,:,:, wInds] = get_patches(ref, patchSize, stride)
        ## preparing features
        deltaViewY = inputView.Y - curRefPos.Y
        deltaViewX = inputView.X - curRefPos.X
        inFeat = prepare_depth_features(curInputLF, deltaViewY, deltaViewX)
        inFeat = crop_img(inFeat, cropSize)
        pInFeat[:,:,:, wInds] = get_patches(inFeat, patchSize, stride)

        ## preparing ref positions
        refPos[0, wInds] =np.tile(curRefPos.Y, (1, numPatches))
        refPos[1, wInds] =np.tile(curRefPos.X, (1, numPatches))

       # print(np.tile('\b', (1, 5)))
        print('Done\n')
    return pInImgs, pInFeat, pRef, refPos
def compute_test_examples(curFullLF, curInputLF):


    #########preparing input images
    [height,width,_,_,_]=curInputLF.shape
    inImgs=curInputLF.reshape((height,width,-1))



    curRefPos = type('', (), {})()
    curRefInd = type('', (), {})()
    curRefInd.Y = 4
    curRefInd.X = 4
    curRefPos.Y = get_img_pos(curRefInd.Y)
    curRefPos.X = get_img_pos(curRefInd.X)

    print('Working on reference (5, 5): ')

    #preparing reference
    ref = curFullLF[:,:,:, curRefInd.Y, curRefInd.X]

    ## preparing features
    deltaViewY = inputView.Y - curRefPos.Y
    deltaViewX = inputView.X - curRefPos.X
    print(deltaViewX)
    print(deltaViewY)
    inFeat = prepare_depth_features(curInputLF, deltaViewY, deltaViewX)

    ## preparing ref positions
    refPos = np.array([[curRefPos.Y],[curRefPos.X]])

    print('Done\n')
    return inImgs, inFeat, ref, refPos


def write_training_examples(inImgs, inFeat, ref, refPos, outputDir, writeOrder, startInd, createFlag, arraySize):
    chunkSize = 1000
    fileName =outputDir+'/training.h5'
    numElements = refPos.shape[1]
    for k in range(0, numElements):
        j = k + startInd
        curInImgs = inImgs[:,:,:, k]
        curInFeat = inFeat[:,:,:, k]
        curRef = ref[:,:,:, k]
        curRefPos = refPos[:, k]
        save_hdf(fileName, 'IN', curInImgs.astype('float32'), pad_with_one(curInImgs.shape, 4), [0, 0, 0, writeOrder[j]], chunkSize, createFlag, arraySize)
        save_hdf(fileName, 'FT', curInFeat.astype('float32'), pad_with_one(curInFeat.shape, 4), [0, 0, 0, writeOrder[j]], chunkSize,
                createFlag, arraySize)
        save_hdf(fileName, 'GT', curRef.astype('float32'), pad_with_one(curRef.shape, 4), [0, 0, 0, writeOrder[j]], chunkSize,
                createFlag, arraySize)
        save_hdf(fileName, 'RP', curRefPos.astype('float32'), curRefPos.shape, [0, writeOrder[j]], chunkSize, createFlag,
                arraySize)

        createFlag = False
    return createFlag


def write_test_examples(inImgs, inFeat, ref, refPos, outputDir):
    chunkSize = 10
    fileName =outputDir+'.h5'
    save_hdf(fileName, 'IN', inImgs.astype('float32'), pad_with_one(inImgs.shape, 4), [0, 0, 0,0], chunkSize, True)
    save_hdf(fileName, 'FT', inFeat.astype('float32'), pad_with_one(inFeat.shape, 4), [0, 0, 0,0], chunkSize,
            True)
    save_hdf(fileName, 'GT', ref.astype('float32'), pad_with_one(ref.shape, 4), [0, 0, 0, 0], chunkSize,
            True)
    save_hdf(fileName, 'RP', refPos.astype('float32'), refPos.shape, [0, 0], chunkSize, True)


def prepare_training_data():
    sceneFolder = param.trainingScenes
    outputFolder = param.trainingData
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder, '.png')
    numPatches = get_num_patches()
    numTotalPatches = numPatches * param.numRefs * numScenes
    writeOrder = np.random.permutation(numTotalPatches)
    firstBatch = True
    make_dir(outputFolder)

    for ns in range(0, numScenes):
        print('**********************************\n')
        print('Working on the "%s" dataset (%d of %d)\n' % (sceneNames[ns][0:- 4], ns, numScenes))

        print('Loading input light field ...')
        [curFullLF, curInputLF] = read_illum_images(scenePaths[ns])
        # print(repmat('\b', 1, 3))
        print('Done\n')
        print('**********************************\n')

        print('\nPreparing training examples\n')
        print('------------------------------\n')
        [pInImgs, pInFeat, pRef, refPos] = compute_training_examples(curFullLF, curInputLF)

        print('\nWriting training examples\n\n')
        firstBatch = write_training_examples(pInImgs, pInFeat, pRef, refPos, outputFolder, writeOrder,
                                             ns * numPatches * param.numRefs, firstBatch, numTotalPatches)


def prepare_test_data():
    sceneFolder = param.testScenes
    outputFolder = param.testData
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder, '.png')

    for ns in range(0, numScenes):
        curOutputName = outputFolder+ '/'+ sceneNames[ns][0: - 4]

        print('**********************************\n')
        print('Working on the "%s" dataset (%d of %d)\n' % (sceneNames[ns][0:- 4], ns, numScenes))

        print('Loading input light field ...')
        [curFullLF, curInputLF] = read_illum_images(scenePaths[ns])
        # print(repmat('\b', 1, 3))
        print('Done\n')
        print('**********************************\n')

        print('\nPreparing test examples\n')
        print('------------------------------\n')
        [pInImgs, pInFeat, pRef, refPos] = compute_test_examples(curFullLF, curInputLF)

        print('\nWriting test examples\n\n')
        write_test_examples(pInImgs, pInFeat, pRef, refPos, curOutputName)

