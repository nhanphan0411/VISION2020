import tensorflow as tf
import numpy as np
import os
import pathlib
import zipfile
from glob import glob
from sklearn.model_selection import train_test_split

from preprocess import ImagePreprocess

class DataLoader(object)
    def __init__(self, 
                 dataset_name='div2k', 
                 scale=2, 
                 subset='train',
                 downgrade='bicubic',
                 ext='png'
                 mode='rgb'
                 images_dir='../data'):
        
        self.dataset_name = dataset_name

        self._ntire_2018 = True

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError('Scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

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

        self.ext = ext
        self.mode = mode
        self.subset = subset
        self.images_dir = images_dir

        os.makedirs(images_dir, exist_ok=True)
    
    def download_div2k(self):
        ''' Download DIV2K dataset to designated directory.
        ''' 
        div2k_img_dir = os.path.join(self.image_dir, 'div2k/images')
        
        hr_download_path = 'DIV2K_{}_HR.zip'.format(self.subset)
        _hr_image_dir = os.path.join(div2k_img_dir, 'DIV2K_{}_HR'.format(self.subset))
        
        if not os.path.exists(self._hr_images_dir()):
            print('Start downloading HR dataset...')
            download_archive(self._hr_images_archive(), div2k_img_dir, extract=True)
        else:
            raise ValueError('Dataset already existed.')

        if self._ntire_2018:
            lr_download_path = 'DIV2K_{}_LR_{}.zip'.format(self.subset, self.downgrade)
            _lr_image_dir = os.path.join(div2k_img_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade))
        else:
            lr_download_path = 'DIV2K_{}_LR_{}_X{}.zip'.format(self.subset, self.downgrade, self.scale)
            _lr_image_dir = return os.path.join(div2k_img_dir, 'DIV2K_{}_LR_{}'.format(self.subset, self.downgrade), 'X{}'.format(self.scale))
        
        if not os.path.exists(self._lr_images_dir()):
            print('Start downloading LR dataset...')
            download_archive(self._lr_images_archive(), div2k_img_dir, extract=True)
        else:
            raise ValueError('Dataset already existed.')

    def load_paths_from_directory(self, directory):
        return sorted(glob(os.path.join(directory, "*.{}".format(self.ext))))
    
    def load_image_from_file(self, path, normalize = False, norm_scale = "0,1"):
        """ loading an image from a file path
        :return: np.array. rgb image
        """
        assert self.mode in ('rgb', 'bgr', 'grayscale')
        assert norm_scale in ("0,1", "-1,1")

        img = cv2.imread(fn, cv2.IMREAD_COLOR if not self.mode == "grayscale" else cv2.IMREAD_GRAYSCALE)

        if mode == "rgb":
            img = img[::-1]

        if normalize:
            if norm_scale == "0,1":
                img /= 255.
            else:
                img = (img / 127.5) - 1.

        return img

    def load_images_from_directory(self, path, normalize = False, norm_scale = "0,1"):
        """ return loaded images from a directory
        :param path: str. file path
        :param ext: str. extension
        :param mode: str. mode to load image
        :param normalize: bool.
        :param norm_scale: str. range of image pixel value to normalize
        :return: list of np.array.
        """
        assert mode in ('rgb', 'grayscale')
        assert norm_scale in ("0,1", "-1,1")

        _files = glob(os.path.join(path, "*.{}".format(self.ext)))

        images = np.asarray([self.load_image_from_file(_file, self.mode, normalize, norm_scale)
                             for _file in tqdm(_files)])
        return images
    
    def load_path_list(self):
        if self.dataset_name == 'div2k':
            hr_train_paths = load_paths_from_directory(div2k_hr_train_dir)
            lr_train_paths = load_paths_from_directory(div2k_lr_train_dir)
            assert len(hr_train_paths) == len(lr_train_paths)
            num_train = len(hr_train_paths)

            hr_test_paths = load_paths_from_directory(div2k_hr_test_dir)
            lr_test_paths = load_paths_from_directory(div2k_lr_test_dir)
            assert len(hr_test_paths) == len(lr_test_paths)
            num_test = len(hr_test_paths)
        
        elif self.dataset_name == 'f2k':
            hr_f2k_paths = load_paths_from_directory(f2k_hr_dir)
            lr_f2k_paths = load_paths_from_directory(f2k_lr_dir)
            assert len(hr_f2k_paths) == len(lr_f2k_paths)

            hr_train_paths, hr_test_paths, lr_train_paths, lr_test_paths = train_test_split(hr_f2k_paths, lr_f2k_paths, test_size = 0.05, random_state=101)
            num_train = len(hr_train_paths)
            num_test = len(hr_test_paths)
    
        return hr_train_paths, lr_train_paths, hr_test_paths, lr_test_paths, num_train, num_test
    
    def load_dataset(self, batch_size=16, repeat_count=None, augment=True):
        hr_train_paths, lr_train_paths, hr_test_paths, lr_test_paths, num_train, num_test = self.load_path_list()
        
        train_ds = tf.data.Dataset.from_tensor_slices((lr_train_paths, hr_train_paths))
        train_ds = ds.map(ImagePreprocess().load_and_preprocess(), num_parallel_calls=AUTOTUNE)
        train_ds = ds.batch(batch_size).repeat(repeat_count).prefetch(buffer_size=AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((lr_test_paths, hr_test_paths))
        val_ds = ds.map(ImagePreprocess().load_and_preprocess(train=False), num_parallel_calls=AUTOTUNE)
        val_ds = ds.batch(batch_size).repeat(repeat_count).prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds, num_train, num_test

# -----------------------------------------------------------
#  Downloader
# -----------------------------------------------------------
def download_archive(file, target_dir, extract=True):
    source_url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/{}'.format(file)
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file)) 

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
            DatasetLoader(scale=4, subset='train', downgrade='bicubic').download_div2k()
            DatasetLoader(scale=4, subset='valid', downgrade='bicubic').download_div2k()
            print('FINISHED.') 