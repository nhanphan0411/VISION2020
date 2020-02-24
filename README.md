# SRGANS

## PREPARE DATASET
To prepare dataset for training, move to ```/src``` and run:

```
python3 data.py --transform --augment [True or False]
```

The DIV2K dataset will be downloaded to ```data``` folder, which locates at the project's root folder. 

(More information about DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/)

After the download process finished, the dataset is transformed into Tensorflow dataset and ready for training. Optional augmentation can be applied by choosing ```True``` or ```False```.

------

## TRAIN

To train the Generator network, run:
```
python3 train.py --type generator --step [number of training step] --evaluate [number of steps]
```

To train the whole GANS network, run:
```
python3 train.py --type gans --step [number of training step] --evaluate [number of steps]
```

```--step``` to indicate number of training steps 

```--evaluate``` to indicate the number of steps that after that losses are updated in log file

- After being trained, weights will be saved at log directory ```../log/ckpt/``` as ```pre_generator.h5``` and ```gan_generator.h5```
- A loss monitor file will also be saved at ```../log```. 

- Note that the training process on the whole GANS network will eventually load the pre-trained weights (```pre_generator.h5```) onto Generator model before starting to train the whole system. 

