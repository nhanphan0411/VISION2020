import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, MaxPool2D
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from tensorflow.python.keras.applications.vgg19 import VGG19


# -----------------------------------------------------------
#  BUILD GENERATOR 
# -----------------------------------------------------------
class Generator(object):
    
    def __init__(self):
        self.n_filters = 64
        self.inc_filter = 32
        self.n_RRDB_blocks = 16
        self.residual_scaling = 0.2
        self.init_kernel = tf.initializers.he_normal(seed=None)
    
    def pixel_shuffle(self, scale):
        return lambda x: tf.nn.depth_to_space(x, scale)
    
    def upsample(self, x_in):
        x = Conv2D(self.n_filters, kernel_size=3, kernel_initializer=self.init_kernel, padding='same')(x_in)
        x = Lambda(self.pixel_shuffle(scale=2))(x)
        x = PReLU(shared_axes=[1, 2])(x)
        
        return x
    
    def conv_RRDB(self, x_in, out_channel, activate=True):
        x = Conv2D(out_channel, kernel_size=3, strides=1, padding='same', kernel_initializer=self.init_kernel)(x_in)
        if activate:
            x = PReLU(shared_axes=[1, 2])(x)
        return x 
    
    def dense_block(self, x_in):
        x = self.conv_RRDB(x_in, self.inc_filter)
        x = self.conv_RRDB(x, self.inc_filter)
        x = self.conv_RRDB(x, self.inc_filter)
        x = self.conv_RRDB(x, self.inc_filter)
        x = self.conv_RRDB(x, self.n_filters, activate=False)
        
        return x * self.residual_scaling
    
    def RRDB(self, x_in):
        x = x_in      
        
        x = self.dense_block(x)
        x = self.dense_block(x)
        x = self.dense_block(x)
        
        x = Add()([x_in, x * self.residual_scaling])
        
        return x 
    
    def build(self, noise_shape):
        x_in = Input(shape=noise_shape)
        
        x = Conv2D(self.n_filters, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, padding='same')(x_in)
        for _ in range(self.n_RRDB_blocks):
            x = self.RRDB(x)
        
        x = Conv2D(self.n_filters, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, padding='same')(x) 
        
        x = self.upsample(x)
        x = self.upsample(x)
        
        x = Conv2D(self.n_filters, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, padding='same')(x) 
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(3, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, padding='same', activation='tanh')(x)
        
        x = Model(x_in, x)
        
        return x


# -----------------------------------------------------------
#  BUILD DISCRIMINATOR
# -----------------------------------------------------------
class Discriminator(object):
    
    def __init__(self):
        self.n_filters = 64
        self.inc_filters = 32
        self.init_kernel = tf.initializers.he_normal(seed=None)
    
    def discriminator_block(self, x_in, out_channel, num=None):
        x = Conv2D(out_channel, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, padding='same')(x_in)
        x = LeakyReLU(alpha=0.2)(x)
    
        x = Conv2D(out_channel, kernel_size=4, strides=2, kernel_initializer=self.init_kernel, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        return x
    
    def build(self, hr_shape):
        x_in = Input(shape=hr_shape)
        
        x = self.discriminator_block(x_in, self.n_filters)
        x = self.discriminator_block(x, self.n_filters * 2)
        x = self.discriminator_block(x, self.n_filters * 4)
        x = self.discriminator_block(x, self.n_filters * 8)
        x = self.discriminator_block(x, self.n_filters * 16)
        
        x = Conv2D(self.n_filters * 16, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, use_bias=False, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(self.n_filters * 16, kernel_size=3, strides=1, kernel_initializer=self.init_kernel, use_bias=False, padding='same')(x)
        
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)

        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1)(x)
        x = Model(x_in, x)
        
        return x

# -----------------------------------------------------------
#  BUILD PRE-TRAINED VGG MODEL
# -----------------------------------------------------------
vgg_19 = VGG19(input_shape=(None, None, 3), weights='imagenet', include_top=False)

def _vgg(output_layer):
    return Model(vgg_19.input, vgg_19.layers[output_layer].output)

def vgg_22():
    return _vgg(5)

def vgg_54():
    return _vgg(20)
