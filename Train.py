import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from Net import depthNetModel, colorNetModel
from PrepareData import *
import warnings

warnings.filterwarnings("ignore")
import h5py
import matplotlib.pyplot as plt


def load_networks(isTraining=False):
    depthNet = depthNetModel()
    colorNet = colorNetModel()

    depthOptimizer = optim.Adam(depthNet.parameters(), lr=param.alpha, betas=(param.beta1, param.beta2), eps=param.eps)
    colorOptimizer = optim.Adam(colorNet.parameters(), lr=param.alpha, betas=(param.beta1, param.beta2), eps=param.eps)

    if isTraining:
        netFolder = param.trainNet
        netName, _, _ = get_folder_content(netFolder, '.tar')

        if param.isContinue and netName:
            tokens = netName[0].split('-')[1].split('.')[0]
            param.startIter = int(tokens)
            checkpoint = torch.load(netFolder + '/' + netName[0])
            depthNet.load_state_dict(checkpoint['depthNet'])
            colorNet.load_state_dict(checkpoint['colorNet'])
            depthOptimizer.load_state_dict(checkpoint['depthOptimizer'])
            colorOptimizer.load_state_dict(checkpoint['colorOptimizer'])
        else:
            param.isContinue = False

    else:
        netFolder = param.testNet
        checkpoint = torch.load(netFolder + '/Net.tar')
        depthNet.load_state_dict(checkpoint['depthNet'])
        colorNet.load_state_dict(checkpoint['colorNet'])
        depthOptimizer.load_state_dict(checkpoint['depthOptimizer'])
        colorOptimizer.load_state_dict(checkpoint['colorOptimizer'])

    return depthNet, colorNet, depthOptimizer, colorOptimizer


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
    maxNumPatches = f[fileInfo[0]].shape[-1]
    numImages = floor(maxNumPatches / batchSize) * batchSize

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
            # wrap them in Variable
            if useGPU:
                features = features.cuda()
            else:
                features = features

        if dataName == 'GT':
            s = f[dataName].shape
            reference = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd + batchSize]
            # print(reference[0, 0, 0, :])
            reference = crop_img(reference, depthBorder + colorBorder)
            reference = torch.from_numpy(reference)
            # wrap them in Variable
            if useGPU:
                reference = reference.cuda()
            else:
                reference = reference

        if dataName == 'IN':
            s = f[dataName].shape
            images = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd + batchSize]
            images = torch.from_numpy(images)
            # wrap them in Variable
            if useGPU:
                images = images.cuda()
            else:
                images = images

        if dataName == 'RP':
            refPos = f[dataName][0:2, startInd:startInd + batchSize]
            refPos = torch.from_numpy(refPos)
            # wrap them in Variable
            if useGPU:
                refPos = refPos .cuda()
            else:
                refPos = refPos

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


def evaluate_system(depthNet, colorNet, depthOptimizer=None, colorOptimizer=None, criterion=None, images=None,
                    refPos=None, isTraining=False, depthFeatures=None, reference=None, isTestDuringTraining=False):
    # Estimating the depth (section 3.1)
    if not isTraining:
        print("Estimating depth\n")
        print("----------------\n")
        print("Extracting depth features")
        dfTime = time.time()
        deltaY = inputView.Y - refPos[0]
        deltaX = inputView.X - refPos[1]
        depthFeatures = prepare_depth_features(images, deltaY, deltaX)
        depthFeatures = np.expand_dims(depthFeatures, axis=3)
        depthFeatures = torch.from_numpy(depthFeatures).float()
        # if param.useGPU:
        #     depthFeatures = Variable(depthFeatures.cuda())
        # else:
        #     depthFeatures = Variable(depthFeatures)

        print('Done in {:.0f} seconds\n'.format(time.time() - dfTime))
    if not isTraining:
        print('Evaluating depth network ...')
        dTime = time.time()
    depthFeatures = depthFeatures.permute(3, 2, 0, 1)  # todo
    depthFeatures = Variable(depthFeatures, requires_grad=True)
    depthRes = depthNet(depthFeatures)
    depth = depthRes / (param.origAngRes - 1)
    depth = depth.data
    depth = depth.permute(2, 3, 1, 0)  # todo
    if not isTraining:
        print('Done in {:.0f} seconds\n'.format(time.time() - dTime))

    # Estimating the final color (section 3.2)
    if not isTraining:
        print("Preparing color features ...")
        cfTime = time.time()

        images = images.reshape((images.shape[0], images.shape[1], -1))
        images = np.expand_dims(images, axis=3)
        images = torch.from_numpy(images)

    colorFeatures, indNan = prepare_color_features(depth, images, refPos)

    if not isTraining:
        print('Done in {:.0f} seconds\n'.format(time.time() - cfTime))

    if not isTraining:
        print('Evaluating color network ...')
        cfTime = time.time()
    colorFeatures = colorFeatures.permute(3, 2, 0, 1)  # todo
    colorFeatures = Variable(colorFeatures, requires_grad=True)
    colorRes = colorNet(colorFeatures)

    finalImg = colorRes
    finalImg = np.transpose(finalImg.data.numpy(), (2, 3, 1, 0))

    if not isTraining:
        print('Done in {:.0f} seconds\n'.format(time.time() - cfTime))
    # Backpropagation
    if isTraining and not isTestDuringTraining:
        loss = criterion(colorRes, Variable(np.transpose(reference, (3, 2, 0, 1)))) / reference.numpy().size

        depthOptimizer.zero_grad()
        colorOptimizer.zero_grad()

        loss.backward(torch.ones(10, 3, 36, 36))

        dzdx = colorFeatures.grad
        dzdx = dzdx.data.permute(2, 3, 1, 0)
        dzdx = prepare_color_features_grad(depth, images, refPos, colorFeatures, indNan, dzdx)
        dzdx = torch.unsqueeze(dzdx, 2)
        dzdx = dzdx.permute(3, 2, 0, 1)
        dzdx = dzdx / (param.origAngRes - 1)

        depthRes.backward(dzdx)

        colorOptimizer.step()
        depthOptimizer.step()

    return finalImg


def compute_psnr(input, ref):
    numPixels = input.size
    sqrdErr = np.sum((input[:] - ref[:]) ** 2) / numPixels
    errEst = 10 * np.log10(1 / sqrdErr)
    return errEst


def test_during_training(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion):
    sceneNames = param.testNames
    fid = open(param.trainNet + '/error.txt', 'a')
    numScenes = len(sceneNames)
    error = 0

    for k in range(numScenes):
        # read input data
        images, depthFeatures, reference, refPos = read_training_data(sceneNames[k], False)
        # evaluate the network and accumulate error
        finalImg = evaluate_system(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion, images, refPos, True,
                                   depthFeatures, reference, True)

        reference = reference.numpy()
        finalImg = crop_img(finalImg, 10)
        reference = crop_img(reference, 10)

        curError = compute_psnr(finalImg, reference)
        error = error + curError / numScenes
    print(error)
    fid.write(str(error) + '\n')
    fid.close()
    return error


def get_test_error(errorFolder):
    testError = []
    if param.isContinue:
        fid = open(errorFolder + '/error.txt', 'r')
        for line in fid:
            testError.append(str(line))
        fid.close()
    else:
        fid = open(errorFolder + '/error.txt', 'w')
        fid.close()
    return testError


def train_system(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion):
    testError = get_test_error(param.trainNet)
    # count=0
    it = param.startIter + 1

    while True:
        it += 1

        if it % param.printInfoIter == 0:
            print('Performing iteration {}'.format(it))

        # main optimization
        depthNet.train(True)  # Set model to training mode
        colorNet.train(True)
        images, depthFeat, reference, refPos = read_training_data(param.trainingNames[0], True, it)
        evaluate_system(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion, images, refPos, True, depthFeat,
                        reference, False)

        if it % param.testNetIter == 0:
            # save network
            _, curNetName, _ = get_folder_content(param.trainNet, '.tar')
            state = {
                'depthNet': depthNet.state_dict(),
                'colorNet': colorNet.state_dict(),
                'depthOptimizer': depthOptimizer.state_dict(),
                'colorOptimizer': colorOptimizer.state_dict()
            }
            torch.save(state, param.trainNet + '/Net-' + str(it) + '.tar')

            # delete network
            if curNetName:
                os.remove(curNetName[0])
            # perform validation
            depthNet.train(False)  # Set model to validation mode
            colorNet.train(False)
            print('\nStarting the validation process\n')
            curError = test_during_training(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion)
            testError.append(curError)
            plt.figure()
            plt.plot(testError)
            plt.title('Current PSNR: %f' % curError)
            plt.show()


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
    [depthNet, colorNet, depthOptimizer, colorOptimizer] = load_networks(True)
    criterion = PairwiseDistance()
    train_system(depthNet, colorNet, depthOptimizer, colorOptimizer, criterion)


if __name__ == "__main__":
    train()
