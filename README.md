# Learning-Based View Synthesis for Light Field Cameras - Pytorch
A PyTorch implementation of a LF Camera View Synthesis method proposed by SIGGRAPH Asia 2016 paper [Learning-Based View Synthesis for Light Field Cameras](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/).
## Requirments
- Python 3.x
- CUDA

## Other dependencies
- Pytorch
- openCV
- scipy
- numpy
- scikit-image
- h5py
```angular2html
pip3 install -r requirments.txt
```
## Datasets

Training and test datasets are from [orginal project page](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/)

### Training Dataset
Training dataset has 100 light field images.
Download the Training set from [here](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/PaperData/SIGGRAPHAsia16_ViewSynthesis_Trainingset.zip),
unzip it and copy the png files in the `TrainingData/Training` directory.

### Test Dataset
Test dataset has 30 light field images.
Download the Test set from [here](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/PaperData/SIGGRAPHAsia16_ViewSynthesis_Testset.zip),
unzip it and copy the png files into `TrainingData/Test`  directory.

## Usage

### Train
Run "PrepareData.m" to process the training and test sets. It takes a long 
time for the first training image to be processed since a huge h5 file needs 
to be created first.
```
python3 prepare_data.py

optional arguments:
--dataset                       choose which dataset to process
``` 
Then start the training
```
python3 train.py

optional arguments:
--is_continue                   if to continue training from existing network[default value is False]
```
The trained network, PSNR value and loss plot are in `TrainingData` directory.

### Test Single Image
Copy desired png files into `Scenes` folder. The results shown in the paper
can be found in `TestSet\PAPER` directory.
```
python3 test.py
```
The output images and objective quality result are in `Results` directory.

## Results

> Original model (iteration: 26,500; Training PSNR: 33.36 ):

- Seahorse (PSNR: 29.10; SSIM: 0.952)

![n_Seahorse](https://thumbs.gfycat.com/AggravatingSomberConey-size_restricted.gif)

- Flower1 (PSNR: 29.86; SSIM: 0.945)

![n_Flower1](https://thumbs.gfycat.com/MixedAstonishingAndeancat-size_restricted.gif)

## To do list:
- Move `prepare_data.py` onto GPU

   This will make training and testing tremendously faster. The key is to implement a cubic `interpolation` method with PyToch tensors to replace the Scipy one.

