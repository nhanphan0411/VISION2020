# **VISION2020 - ESRGAN IMPLEMENTATION ON TENSORFLOW 2**

**The project is by Nhan Phan, as his Final Project for the Machine Learning Bootcamp at CoderSchool (Feb/2020)**

![](https://miro.medium.com/proxy/1*E-JmUwv7zbwjzFm1hJLxPA.png)

![alt text](https://github.com/xinntao/ESRGAN/blob/master/figures/baboon.jpg?raw=true)

**VISION2020** aims at recovering a high resolution image from a low resolution one. The project is based largely on the excellent research of Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, Chen Change Loy on [ESRGAN](https://arxiv.org/pdf/1809.00219v2.pdf) (2018) and their [implementation](https://github.com/xinntao/ESRGAN) using Pytorch.

**Single image super-resolution (SISR)**, as a fundamental low-level vision problem, has attracted increasing attention in the research community and AI companies. SISR aims at recovering a high-resolution (HR) image from a single low-resolution (LR) one. Since the pioneer work of SRCNN proposed by Dong et al., deep convolution neural network (CNN) approaches have brought prosperous development. Various network architecture designs and training strategies have continuously improved the SR performance.

The **Super-Resolution Generative Adversarial Network (SRGAN)** is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an **Enhanced SRGAN (ESRGAN)**. 

In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. 

Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN. The project is optimized and built entirely on Tensorflow 2.0. It successfully resizes the image up to x64 on square area.

Result samples ▸

**x4 per dimension**
<img src='https://i.imgur.com/vzw0TvE.png'>
<img src='https://i.imgur.com/i09Wuk8.png'>

**x8 per dimension**
<img src="https://i.imgur.com/EAPumPH.png">
<img src='https://i.imgur.com/LKvSC7L.png'>


------
<br></br>
## 📖 FOLDER STRUCTURE

```
|--data/
|------div2k/
|------Flickr2K/
|--src/
|--log/
|------model/
|--app/
```
<br></br>
## 📀 PREPARE DATASET
To prepare dataset for training, move to ```/src``` and run:

```
python3 data.py --download --dataset ['div2k' or 'f2k']
```

The dataset will be downloaded to ```data``` folder, which locates at the project's root folder. 

- [More information about DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/): 
- [More information about Flickr2K dataset](https://github.com/limbee/NTIRE2017#dataset)

<br></br>

## 🤖 TRAIN

Training for ESRGAN includes two phases:
- Warm-up training for Generator
- Training for the whole GAN model. 

To start the training session, run: 

```
python3 train.py 
--type ['generator' or 'gan'] 
--dataset ['div2k' or 'f2k']
--genpath [generator's weight path - optional]
--dispath [discriminator's weight path - optional]
--epochs [number of training epochs]
--print_every [print training record after that number of epochs]
--save_every [save model weight after that number of epochs]
--logname [name of loss record log file] 
--modelname [name of saved model]
```

Once the training is done, the model will be saved at ```../log/model```

<br></br>

## ✨ DEMO WEB APP

To try demo web app, go back to project root's folder and run:

```
python3 app/main.py
```

<br></br>

## 🤓 REFERENCES 

- ESRGAN on Tensorflow 1 -- https://github.com/hiram64/ESRGAN-tensorflow
- ESRGAN on PyTorch -- https://github.com/xinntao/ESRGAN
- SRGAN framework on Tensorflow 2 -- https://github.com/krasserm/super-resolution
- Research on ESRGAN -- https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative
- Research on SRGAN -- https://www.paperswithcode.com/paper/photo-realistic-single-image-super-resolution
