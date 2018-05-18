import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
from Net import depthNetModel,colorNetModel
from math import floor
from InitParam import param,novelView,inputView,get_folder_content
from PrepareData import*
import warnings
warnings.filterwarnings("ignore")
import h5py
import re
import matplotlib.pyplot as plt
from Train import load_networks,read_illum_images,evaluate_system,compute_psnr
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import compare_ssim as ssim
from cv2 import imwrite

def adjust_tone(input):

    input[input > 1] = 1
    input[input < 0] = 0
    output = input ** (1/1.5)
    output = rgb2hsv(output)
    output[:, :, 1] = output[:, :, 1] * 1.5
    output = hsv2rgb(output)
    return output

def get_img_ind(inPos):
    ind = int(inPos * (param.origAngRes - 1))
    return ind

def write_error(estimated, reference, resultPath):
    quantizedEst = (estimated * 255).astype(int)
    quantizedRef = (reference * 255).astype(int)
    curPSNR = compute_psnr(estimated, reference)
    curSSIM = ssim(quantizedEst, quantizedRef,multichannel=True)

    fid = open(resultPath+'/ObjectiveQuality.txt', 'w')
    fid.write('PSNR: %3.2f\n' % curPSNR)
    fid.write('SSIM: %1.3f\n' % curSSIM)
    fid.close()

def synthesize_novel_views(depthNet, colorNet, inputLF, fullLF, resultPath):

    numNovelViews = len(novelView.Y)

    # for vi in range(numNovelViews):
    for vi in range(1):
        indY = get_img_ind(novelView.Y[vi])
        indX = get_img_ind(novelView.X[vi])

        curRefPos = np.array([novelView.Y[vi], novelView.X[vi]])
        curRefPos = np.expand_dims(curRefPos,axis=1)
        # performs the whole process of extracting features, evaluating the
        # two sequential networks and generating the output synthesized image
        print('\nView %02d of %02d\n' % (vi, numNovelViews))
        print('**********************************\n')
        synthesizedView = evaluate_system(depthNet, colorNet, images = inputLF, refPos = curRefPos)

        synthesizedView = synthesizedView[:,:,:,-1]
        # crop the result and reference images
        curEst = crop_img(synthesizedView, 10)
        curRef = crop_img(fullLF[:,:,:, indY, indX], param.depthBorder + param.colorBorder + 10)

        # quantize the reference and estimated image to 8 bit for accurate
        # numerical evaluation
        # quantizedEst = (curEst * 255).astype(int)
        # quantizedRef = (curRef * 255).astype(int)

        # write the numerical evaluation and the final image
        if indY == 0 and indX == 0:
            write_error(curEst, curRef, resultPath)
        # print(curEst)
        imwrite(resultPath+'/Images/'+('%02d_%02d.png' % (indY, indX)), (adjust_tone(curEst)*255).astype(int))

def test():
    # Initialization
    sceneFolder = './Scenes'
    resultFolder = './Results'


    # load the pre-trained networks
    [depthNet, colorNet,_,_] = load_networks()


    # Generate novel views for each scene
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder)

    for ns in range(numScenes):
        print('**********************************\n')
        print('Working on the '+sceneNames[ns][0:- 4]+'dataset\n')

        resultPath = resultFolder+ '/'+ sceneNames[ns][0:- 4]
        make_dir( resultPath + '/Images')

        print('Loading input light field ...')
        [curFullLF, curInputLF] = read_illum_images(scenePaths[ns])
        print('Done\n')
        print('**********************************\n')

        print('\nSynthesizing novel views\n')
        print('--------------------------\n')
        synthesize_novel_views(depthNet, colorNet, curInputLF, curFullLF, resultPath)
        print('\n\n\n')

if __name__ == "__main__":
    test()