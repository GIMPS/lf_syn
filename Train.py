import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
from Net import depthNet,colorNet
from math import floor
from InitParam import param,novelView,inputView,get_folder_content
from PrepareData import prepare_depth_features,prepare_color_features,prepare_training_data,prepare_test_data
import warnings
warnings.filterwarnings("ignore")
import h5py

def load_networks(isTraining):
    model_ft.load_state_dict(torch.load('trained_nn'))
def train_system():
    torch.save(state, 'trained_state')

use_gpu = torch.cuda.is_available()




def read_training_data(fileName, isTraining, it):
    batchSize = param.batchSize
    depthBorder = param.depthBorder
    colorBorder = param.colorBorder
    useGPU = param.useGPU

    f = h5py.File(fileName, "r")
    fileInfo=[]
    for item in f.keys():
        fileInfo.append(item)
    numItems = len(fileInfo)
    maxNumPatches = f[fileInfo[0]].shape[-1]
    numImages = floor(maxNumPatches / batchSize) * batchSize

    if isTraining:
        startInd = (it - 1) * batchSize % numImages + 1
    else:
        startInd = 1
        batchSize = 1

    features = []
    reference = []
    images = []
    refPos = []

    for i in range(numItems):
        dataName = fileInfo[i]

        if dataName == 'FT':
            s = f[fileInfo[0]].shape
            features = h5read(fileName, '/FT', [0, 0, 0, startInd], [s[0], s[1], s[2], batchSize])
            features = single(features)
            if useGPU:
                features = gpuArray(features)

        if dataName == 'GT':
            s = fileInfo.Datasets(i).Dataspace.Size;
            reference = h5read(fileName, '/GT', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize])
            reference = single(CropImg(reference, depthBorder+colorBorder));
            if useGPU:
                reference = gpuArray(reference)

        if dataName == 'IN':
            s = fileInfo.Datasets(i).Dataspace.Size;
            images = h5read(fileName, '/IN', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize])
            if useGPU:
                images = gpuArray(images)

        if dataName == 'RP':
            refPos = h5read(fileName, '/RP', [1, startInd], [2, batchSize])
            refPos = single(refPos)
            if useGPU:
                refPos = gpuArray(refPos)
    f .close()
    return images, features, reference, refPos




def evaluate_system(depthNet, colorNet, images, refPos, isTraining, depthFeatures, reference, isTestDuringTraining):
    if not isTraining:
        print("Estimating depth\n")
        print("----------------\n")
        print("Extracting depth features")
        deltaY = inputView.Y - refPos[0,:]
        deltaX = inputView.X - refPos[1,:]
        depthFeatures =prepare_depth_features(images, deltaY, deltaX)
        #todo
        # if (param.useGPU)
        #     depthFeatures = gpuArray(depthFeatures);
        # end

    depthRes = EvaluateNet(depthNet, depthFeatures, [], true)
    depth = depthRes(end).x / (param.origAngRes - 1)


    if not isTraining:
        print("Preparing color features")
        images=images.reshape((images.shape[0], images.shape[1], -1))
    [colorFeatures, indNan] = prepare_color_features(depth, images, refPos)

    colorRes = EvaluateNet(colorNet, colorFeatures, [], true)
    finalImg = colorRes[-1].x


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch =0
    while True:
        epoch+=1
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                if model.training == True:
                    # forward
                    outputs, aux_output = model(inputs)
                else:
                    outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            all_losses.append(epoch_loss)
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'trained_nn')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train():
    depthModel = depthNet()

    if use_gpu:
        depthModel = depthModel.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(depthModel.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(depthModel, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)


if __name__ == "__main__":
    fileName="./TrainingData/Test/Cars.h5"
    # prepare_test_data()
    #prepare_training_data()
    batchSize = param.batchSize
    depthBorder = param.depthBorder
    colorBorder = param.colorBorder
    useGPU = param.useGPU

    f = h5py.File(fileName, "r")
    fileInfo=[]
    for item in f.keys():
        fileInfo.append(item)
    numItems = len(fileInfo)
    maxNumPatches = f[fileInfo[0]].shape[-1]
    numImages = floor(maxNumPatches / batchSize) * batchSize

    if False:
        startInd = (1 - 1) * batchSize % numImages
    else:
        startInd = 0
        batchSize = 1

    features = []
    reference = []
    images = []
    refPos = []
    dataName = fileInfo[0]
    if dataName == 'FT':
        s = f[fileInfo[0]].shape
        features = f[dataName][0:s[0], 0:s[1], 0:s[2], startInd:startInd+batchSize]
        print(features)
