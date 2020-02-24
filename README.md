# ESRGANS

The Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) is a seminal work that is capable of upscaling low-resolution images to high-resolution ones while generating realistic textures. The whole project is built on Tensorflow2. Its model is designed based on Xintao Wang et al’s research on ESRGAN (2018) - inherits the idea about Residual-in-Residual Dense Block. 

At the end, the project successfully generates high-resolution photo that sizes up to 8 times (per dimension) compared with the low-resolution one. 

This repository is still a work-in-process. 

## PREPARE DATASET
To prepare dataset for training, move to ```/src``` and run:

```
python3 data.py --download --augment div2k
```

The DIV2K dataset will be downloaded to ```data``` folder, which locates at the project's root folder. 

(More information about DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/)

After the download process finished, the dataset is transformed into Tensorflow dataset and ready for training. Optional augmentation can be applied by choosing ```True``` or ```False```.

------

## TRAIN

To train the Generator network, run:
```
python3 train.py --train --type generator --epochs [number of training epoch] --logname [name of trainning session log file]
```

To train the whole GANS network, run:
```
python3 train.py --train --type gan --epochs [number of training epoch] --logname [name of trainning session log file]
```

