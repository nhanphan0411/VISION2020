## ✺ ABOUT

VISION2020 is a work to implement ESRGAN in order to improve image resolution. This is my final project for Machine Learning Bootcamp (Mariana Class) at CoderSchool (Feb/2020) 

ESRGAN - The Enhanced Super-Resolution Generative Adversarial Network is a seminal work that is capable of upscaling low-resolution images to high-resolution ones while generating realistic textures. The whole project is built on Tensorflow2. Its model is designed based on Xintao Wang et al’s research on ESRGAN (2018) - inherits the idea about Residual-in-Residual Dense Block. 

At the end, the project successfully generates high-resolution photo that sizes up to 8 times (per dimension) compared with the low-resolution one. 

------
## ⌘ DATA STRUCTURE

```
|--data/
|----div2k/
|----Flickr2K/
|--src/
|--log/
|----model/
|--app/
```

------

Please refer to below code to download the dataset. 

## ▼ PREPARE DATASET
To prepare dataset for training, move to ```/src``` and run:

```
python3 data.py --download --dataset ['div2k' or 'f2k']
```

The dataset will be downloaded to ```data``` folder, which locates at the project's root folder. 

(More information about DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/)
(More information about Flickr2K dataset: https://github.com/limbee/NTIRE2017#dataset)

------

## ☺︎ TRAIN

Training process for the GAN system includes two phases Initial Generator Training and GAN Training. To start the training session, run: 

```
python3 train.py 
--type ['generator' or 'gan'] 
--dataset div2k 
--epochs [number of training epochs]
--genpath [generator's weight path - optional]
--dispath [discriminator's weight path - optional]
--logname [name of loss record log file] 
--modelname [name of saved model]
```

Saved model directory is set at '../log/model' at default.

-------

## ► DEMO WEB APP

To run demo app, back to project root's folder and run:

```
python3 app/main.py
```

-------

## ◉ REFERENCES 

- ESRGAN on Tensorflow 1 -- https://github.com/hiram64/ESRGAN-tensorflow
- ESRGAN on PyTorch -- https://github.com/xinntao/ESRGAN
- SRGAN framework on Tensorflow 2 -- https://github.com/krasserm/super-resolution

- Research on ESRGAN -- https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative
- Research on SRGAN -- https://www.paperswithcode.com/paper/photo-realistic-single-image-super-resolution
