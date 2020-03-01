import tensorflow as tf
import numpy as np
import os
import pathlib
import zipfile
from glob import glob
from sklearn.model_selection import train_test_split
import argparse
from preprocess import ImagePreprocess
from tensorflow.python.data.experimental import AUTOTUNE
from model import *

class DataLoader(object):
    def __init__(self, 
                 dataset_name='div2k', 
                 ext='png',
                 mode='rgb',
                 channels=3,
                 scale = 4,
                 subset = 'train',
                 downgrade = 'bicubic',
                 image_dir='../data'):
        
        self.dataset_name = dataset_name
        self.ext = ext
        self.mode = mode
        self.channels = channels
        self.image_dir = image_dir

        self._ntire_2018 = True
        
        _scales = [2, 3, 4, 8]
        if scale not in _scales:
            raise ValueError('Scale must be in ${_scales}')
        else:
            self.scale = scale
        
        _subsets = ['train', 'valid']
        if subset not in _subsets:
            raise ValueError("subset must be 'train' or 'valid'")
        else: 
            self.subset = subset

        _downgrades_a = ['bicubic', 'unknown']
        _downgrades_b = ['mild', 'difficult']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError('Scale 8 only allowed for bicubic downgrade.')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError('{} downgrade requires scale 4'.format(downgrade))

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
        else:
            self.downgrade = downgrade
            self._ntire_2018 = False

        os.makedirs(image_dir, exist_ok=True)
    
    # --------// DOWNLOAD MODULE // --------
    def download_div2k(self):
        ''' Download DIV2K dataset to designated directory.
        '''
        source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/'
        div2k_img_dir = os.path.join(self.image_dir, 'div2k/images')
        
        # DOWNLOAD HR DATASET
        hr_file = 'DIV2K_{}_HR.zip'.format(self.subset)
        _hr_image_dir = os.path.join(div2k_img_dir, 'DIV2K_{}_HR'.format(self.subset))
        
        if not os.path.exists(_hr_image_dir):
            print('Start downloading HR dataset...')
            download_archive(hr_file, source_url, div2k_img_dir, extract=True)
        else:
            raise ValueError('Dataset already existed.')
        
        # DOWNLOAD LR DATASET
        if self._ntire_2018:
            lr_file = 'DIV2K_{}_LR_{}.zip'.format(self.subset, self.downgrade)
            _lr_image_dir = os.path.join(div2k_img_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade))
        else:
            lr_file = 'DIV2K_{}_LR_{}_X{}.zip'.format(self.subset, self.downgrade, self.scale)
            _lr_image_dir = os.path.join(div2k_img_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade), 'X{}'.format(self.scale))
        
        if not os.path.exists(_lr_image_dir):
            print('Start downloading LR dataset...')
            download_archive(lr_file, source_url, div2k_img_dir, extract=True)
        else:
            raise ValueError('Dataset already existed.')

    def download_f2k(self):
        source_url = 'http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar'
        f2k_file = 'Flickr2K.tar'
        f2k_img_dir = os.path.join(self.image_dir, 'Flickr2K')
        if not os.path.exists(f2k_img_dir):
            download_archive(f2k_file, source_url, f2k_img_dir, extract=True)

    # --------// DATA LOADING MODULE // --------
    def load_paths_from_directory(self, directory):
        return sorted(glob(os.path.join(directory, "*.{}".format(self.ext))))
    
    def load_path_list(self):
        if self.dataset_name == 'div2k':
            lr_dir = os.path.join(self.image_dir, 'div2k/images/DIV2K_{}_LR_bicubic/X{}'.format(self.subset, self.scale))
            hr_dir = os.path.join(self.image_dir, 'div2k/images/DIV2K_{}_HR'.format(self.subset))
            
        elif self.dataset_name == 'f2k':
            lr_dir = os.path.join(self.image_dir, 'Flickr2K/Flickr2K/Flickr2K_LR_bicubic')
            hr_dir = os.path.join(self.image_dir, 'Flickr2K/Flickr2K/Flickr2K_HR')

        lr_paths = self.load_paths_from_directory(lr_dir)
        print(lr_paths[0])
        hr_paths = self.load_paths_from_directory(hr_dir)
        print(hr_paths[0])
        assert len(lr_paths) == len(hr_paths)
        num_data = len(hr_paths)

        return lr_paths, hr_paths, num_data
    
    def load_and_preprocess(self, lr_path, hr_path, augment=False):
        lr = tf.io.read_file(lr_path)
        lr = tf.image.decode_png(lr, channels=self.channels)

        hr = tf.io.read_file(hr_path)
        hr = tf.image.decode_png(hr, channels=self.channels)
    
        if self.subset == 'train': 
            lr, hr = random_crop(lr, hr)
            if augment == True:
                lr, hr = random_flip(lr, hr)
                lr, hr = random_rotate(lr, hr)
        
        lr = tf.cast(lr, dtype=tf.float32) / 255.
        hr = tf.cast(hr, dtype=tf.float32) / 255.

        return lr, hr
    
    def load_dataset(self, batch_size=1, repeat_count=None):
        lr_paths, hr_paths, num_data = self.load_path_list()
        
        ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
        ds = ds.map(self.load_and_preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).repeat(repeat_count).prefetch(buffer_size=AUTOTUNE)
        
        return ds, num_data

# -----------------------------------------------------------
#  Downloader
# -----------------------------------------------------------

def download_archive(file, source_url, target_dir, extract=True):
    download_path = os.path.join(source_url, file)
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, download_path, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file)) 

# -----------------------------------------------------------
#  Image Transformation 
# -----------------------------------------------------------
def random_crop(x_lr, x_hr, hr_size=128, scale=4):
        lr_size = hr_size // scale
        lr_img_shape = tf.shape(x_lr)[:2]

        lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_size + 1, dtype=tf.int32)
        lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_size + 1, dtype=tf.int32)

        hr_w = lr_w * scale
        hr_h = lr_h * scale

        x_lr_cropped = x_lr[lr_h:lr_h + lr_size, lr_w:lr_w + lr_size]
        x_hr_cropped = x_hr[hr_h:hr_h + hr_size, hr_w:hr_w + hr_size]

        return x_lr_cropped, x_hr_cropped
    
def random_flip(x_lr, x_hr):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                lambda: (x_lr, x_hr),
                lambda: (tf.image.flip_left_right(x_lr),
                        tf.image.flip_left_right(x_hr)))

def random_rotate(x_lr, x_hr):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(x_lr, rn), tf.image.rot90(x_hr, rn)

# -----------------------------------------------------------
#  Command
# -----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    if args.download:
        if args.dataset == "div2k":
            print('PREPARING DIV2K DATASET')  
            DataLoader().download_div2k(scale=4, subset='train', downgrade='bicubic')
            DataLoader().download_div2k(scale=4, subset='valid', downgrade='bicubic')
            print('FINISHED.') 

        if args.dataset == "f2k":
            print('PREPARING F2K DATASET')  
            DataLoader(dataset_name='f2k').download_f2k()
            print('FINISHED.') 