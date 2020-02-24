import tensorflow as tf
import numpy as np

class ImagePreprocess(object): 
    def __init__(self, hr_size=128, scale=4):
        self.hr_size = hr_size
        self.scale = scale

    def random_crop(self, x_lr, x_hr, hr_crop_size=self.hr_size):
        lr_crop_size = hr_crop_size // self.scale
        lr_img_shape = tf.shape(x_lr)[:2]

        lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
        lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

        hr_w = lr_w * self.scale
        hr_h = lr_h * self.scale

        x_lr_cropped = x_lr[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        x_hr_cropped = x_hr[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

        return x_lr_cropped, x_hr_cropped
    
    def random_flip(self, x_lr, x_hr):
        rn = tf.random.uniform(shape=(), maxval=1)
        return tf.cond(rn < 0.5,
                   lambda: (x_lr, x_hr),
                   lambda: (tf.image.flip_left_right(x_lr),
                            tf.image.flip_left_right(x_hr)))
    
    def random_rotate(self, x_lr, x_hr):
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(x_lr, rn), tf.image.rot90(x_hr, rn)

    def load_and_preprocess(self, lr_path, hr_path, train=True, augment=True):
        lr = tf.io.read_file(lr_path)
        lr = tf.image.decode_png(lr, channels=self.channels)
        lr = tf.cast(lr, dtype=tf.float32) / 255.

        hr = tf.io.read_file(hr_path)
        hr = tf.image.decode_png(hr, channels=self.channels)
        hr = tf.cast(hr, dtype=tf.float32) / 255.
        
        if train: 
            lr, hr = self.random_crop(lr, hr)
            if self.augment==True:
                lr, hr = self.random_flip(lr, hr)
                lr, hr = self.random_rotate(lr, hr)

        return lr, hr