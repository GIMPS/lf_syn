import time
from math import log10

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import DepthNetModel, ColorNetModel
from prepare_data import *

warnings.filterwarnings("ignore")
import h5py

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--is_continue', default=False, type=bool, help='if to continue training from existing network')
opt = parser.parse_args()
param.isContinue = opt.is_continue


def load_networks(isTraining=False):
    depth_net = DepthNetModel()
    color_net = ColorNetModel()
    if param.useGPU:
        depth_net.cuda()
        color_net.cuda()

    depth_optimizer = optim.Adam(depth_net.parameters(), lr=param.alpha, betas=(param.beta1, param.beta2),
                                 eps=param.eps)
    color_optimizer = optim.Adam(color_net.parameters(), lr=param.alpha, betas=(param.beta1, param.beta2),
                                 eps=param.eps)

    if isTraining:
        netFolder = param.trainNet
        netName, _, _ = get_folder_content(netFolder, '.tar')

        if param.isContinue and netName:
            tokens = netName[0].split('-')[1].split('.')[0]
            param.startIter = int(tokens)
            checkpoint = torch.load(netFolder + '/' + netName[0])
            depth_net.load_state_dict(checkpoint['depth_net'])
            color_net.load_state_dict(checkpoint['color_net'])
            depth_optimizer.load_state_dict(checkpoint['depth_optimizer'])
            color_optimizer.load_state_dict(checkpoint['color_optimizer'])
        else:
            param.isContinue = False

    else:
        netFolder = param.testNet
        checkpoint = torch.load(netFolder + '/Net.tar')
        depth_net.load_state_dict(checkpoint['depth_net'])
        color_net.load_state_dict(checkpoint['color_net'])
        depth_optimizer.load_state_dict(checkpoint['depth_optimizer'])
        color_optimizer.load_state_dict(checkpoint['color_optimizer'])

    return depth_net, color_net, depth_optimizer, color_optimizer


def read_training_data(fileName, isTraining, it=0):
    batchSize = param.batchSize
    depthBorder = param.depthBorder
    colorBorder = param.colorBorder
    useGPU = param.useGPU

    f = h5py.File(fileName, "r")
    fileInfo = []
    for item in f.keys():
        fileInfo.append(item)
    numItems = len(fileInfo)
    maxnum_patches = f[fileInfo[0]].shape[-1]
    numImages = floor(maxnum_patches / batchSize) * batchSize

    if isTraining:
        startInd = it * batchSize % numImages
    else:
        startInd = 0
        batchSize = 1

    features = []
    reference = []
    images = []
    refPos = []

    for i in range(numItems):
        dataName = fileInfo[i]

        if dataName == 'FT':
            s = f[dataName].shape
            features = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd + batchSize]
            features = torch.from_numpy(features)
            if useGPU:
                features = features.cuda()

        if dataName == 'GT':
            s = f[dataName].shape
            reference = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd + batchSize]
            reference = crop_img(reference, depthBorder + colorBorder)
            reference = torch.from_numpy(reference)
            if useGPU:
                reference = reference.cuda()

        if dataName == 'IN':
            s = f[dataName].shape
            images = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd + batchSize]
            images = torch.from_numpy(images)
            if useGPU:
                images = images.cuda()

        if dataName == 'RP':
            refPos = f[dataName][0:2, startInd:startInd + batchSize]
            refPos = torch.from_numpy(refPos)
            if useGPU:
                refPos = refPos.cuda()

    f.close()
    return images, features, reference, refPos


def prepare_color_features_grad(depth, images, refPos, curFeatures, indNan, dzdx):
    delta = 0.01
    depthP = depth + delta
    featuresP, indNanP = prepare_color_features(depthP, images, refPos)
    grad = (featuresP - curFeatures.data.permute(2, 3, 1, 0)) / delta * dzdx
    tmp = grad[:, :, 1: - 2, :]
    tmp[indNan | indNanP] = 0
    grad[:, :, 1:- 2, :] = tmp
    dzdx = torch.sum(grad, 2)
    return dzdx


def evaluate_system(depth_net, color_net, depth_optimizer=None, color_optimizer=None, criterion=None, images=None,
                    refPos=None, isTraining=False, depthFeatures=None, reference=None, isTestDuringTraining=False):
    # Estimating the depth (section 3.1)
    if not isTraining:
        print("Estimating depth")
        print("----------------")
        print("Extracting depth features...", end='   ')
        dfTime = time.time()
        deltaY = inputView.Y - refPos[0]
        deltaX = inputView.X - refPos[1]
        depthFeatures = prepare_depth_features(images, deltaY, deltaX)
        depthFeatures = np.expand_dims(depthFeatures, axis=3)
        depthFeatures = torch.from_numpy(depthFeatures).float()
        if param.useGPU:
            depthFeatures = depthFeatures.cuda()

        print('\b\b\b\bDone in {:.0f} seconds'.format(time.time() - dfTime))
    if not isTraining:
        print('Evaluating depth network ...', end='')
        dTime = time.time()
    depthFeatures = depthFeatures.permute(3, 2, 0, 1)
    depthFeatures = Variable(depthFeatures, requires_grad=True)
    depthRes = depth_net(depthFeatures)
    depth = depthRes / (param.origAngRes - 1)
    depth = depth.data
    depth = depth.permute(2, 3, 1, 0)
    if not isTraining:
        print('Done in {:.0f} seconds'.format(time.time() - dTime), flush=True)

    # Estimating the final color (section 3.2)
    if not isTraining:
        print("Preparing color features ...", end='')
        cfTime = time.time()

        images = images.reshape((images.shape[0], images.shape[1], -1))
        images = np.expand_dims(images, axis=3)
        images = torch.from_numpy(images)

    colorFeatures, indNan = prepare_color_features(depth, images, refPos)

    if not isTraining:
        print('Done in {:.0f} seconds'.format(time.time() - cfTime))

    if not isTraining:
        print('Evaluating color network ...', end='')
        cfTime = time.time()
    colorFeatures = colorFeatures.permute(3, 2, 0, 1)
    colorFeatures = Variable(colorFeatures, requires_grad=True)
    colorRes = color_net(colorFeatures)

    finalImg = colorRes
    finalImg = finalImg.data.permute(2, 3, 1, 0)

    if not isTraining:
        print('Done in {:.0f} seconds'.format(time.time() - cfTime))
    # Backpropagation
    if isTraining and not isTestDuringTraining:
        loss = criterion(colorRes, Variable(reference.permute(3, 2, 0, 1))) / reference.cpu().numpy().size

        depth_optimizer.zero_grad()
        color_optimizer.zero_grad()
        if param.useGPU:
            loss.backward(torch.ones(10, 3, 36, 36).cuda())
        else:
            loss.backward(torch.ones(10, 3, 36, 36))

        dzdx = colorFeatures.grad
        dzdx = dzdx.data.permute(2, 3, 1, 0)
        dzdx = prepare_color_features_grad(depth, images, refPos, colorFeatures, indNan, dzdx)
        dzdx = torch.unsqueeze(dzdx, 2)
        dzdx = dzdx.permute(3, 2, 0, 1)
        dzdx = dzdx / (param.origAngRes - 1)

        depthRes.backward(dzdx)

        color_optimizer.step()
        depth_optimizer.step()

    return finalImg


def compute_psnr(input, ref):
    numPixels = input.numel()
    sqrdErr = torch.sum((input[:] - ref[:]) ** 2) / numPixels
    errEst = 10 * log10(1 / sqrdErr)
    return errEst


def test_during_training(depth_net, color_net, depth_optimizer, color_optimizer, criterion):
    sceneNames = param.testNames
    fid = open(param.trainNet + '/error.txt', 'a')
    numScenes = len(sceneNames)
    error = 0

    for k in range(numScenes):
        # read input data
        images, depthFeatures, reference, refPos = read_training_data(sceneNames[k], False)
        # evaluate the network and accumulate error
        finalImg = evaluate_system(depth_net, color_net, depth_optimizer, color_optimizer, criterion, images, refPos,
                                   True,
                                   depthFeatures, reference, True)

        finalImg = crop_img(finalImg, 10)
        reference = crop_img(reference, 10)

        curError = compute_psnr(finalImg, reference)
        error = error + curError / numScenes
    print('Current PSNR: %.3f' % error)
    fid.write(str(error) + '\n')
    fid.close()
    return error


def get_test_error(errorFolder):
    testError = []
    if param.isContinue:
        fid = open(errorFolder + '/error.txt', 'r')
        for line in fid:
            testError.append(float(line))
        fid.close()
    else:
        fid = open(errorFolder + '/error.txt', 'w')
        fid.close()
    return testError


def train_system(depth_net, color_net, depth_optimizer, color_optimizer, criterion):
    testError = get_test_error(param.trainNet)
    it = param.startIter + 1

    while True:
        if it % param.printInfoIter == 0:
            print('Performing iteration {}'.format(it))

        # main optimization
        depth_net.train(True)  # Set model to training mode
        color_net.train(True)
        images, depthFeat, reference, refPos = read_training_data(param.trainingNames[0], True, it)
        evaluate_system(depth_net, color_net, depth_optimizer, color_optimizer, criterion, images, refPos, True,
                        depthFeat,
                        reference, False)

        if it % param.testNetIter == 0:
            # save network
            _, curNetName, _ = get_folder_content(param.trainNet, '.tar')
            state = {
                'depth_net': depth_net.state_dict(),
                'color_net': color_net.state_dict(),
                'depth_optimizer': depth_optimizer.state_dict(),
                'color_optimizer': color_optimizer.state_dict()
            }
            torch.save(state, param.trainNet + '/Net-' + str(it) + '.tar')

            # delete network
            if curNetName:
                os.remove(curNetName[0])
            # perform validation
            depth_net.train(False)  # Set model to validation mode
            color_net.train(False)
            print('Starting the validation process... ', end='', flush=True)
            curError = test_during_training(depth_net, color_net, depth_optimizer, color_optimizer, criterion)
            testError.append(curError)
            # plt.plot(testError)
            # plt.title('Current PSNR: %f' % curError)
            # plt.savefig(param.trainNet + '/fig.png')

        it += 1


class PairwiseDistance(nn.Module):
    def __init__(self, p=2, eps=1e-6):
        super(PairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps

    def forward(self, x1, x2):
        return pairwise_distance(x1, x2, self.norm, self.eps)


def pairwise_distance(x1, x2, p=2, eps=1e-6):
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p)
    return out


def train():
    [depth_net, color_net, depth_optimizer, color_optimizer] = load_networks(True)
    criterion = PairwiseDistance()
    if param.useGPU:
        criterion.cuda()
    train_system(depth_net, color_net, depth_optimizer, color_optimizer, criterion)


if __name__ == "__main__":
    train()
