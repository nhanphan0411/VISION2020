import tensorflow as tf
import numpy as np
import time
import argparse

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean
import tensorflow.keras.backend as K
from tensorflow.python.data.experimental import AUTOTUNE

from model import *
from data import *

# -----------------------------------------------------------
#  CONFIG
# -----------------------------------------------------------
LOG_DIR = '../log'
model_save_dir = '../log/model'
gan_loss_coeff = 0.005
content_loss_coeff = 0.01

# -----------------------------------------------------------
#  LOSSES
# -----------------------------------------------------------
class Loss(object):
    def pretrain_loss(self, pre_gen_out, HR_data):
        pre_gen_loss = tf.reduce_mean(tf.reduce_mean(tf.square(pre_gen_out - HR_data), axis=3))

        return pre_gen_loss

    def train_loss(self, sr, hr, sr_output, hr_output):
        # Gen loss
        g_loss_p1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_output - tf.reduce_mean(sr_output), labels=tf.zeros_like(hr_output)))
        g_loss_p2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sr_output - tf.reduce_mean(hr_output), labels=tf.ones_like(sr_output)))
        generator_loss = gan_loss_coeff * (g_loss_p1 + g_loss_p2) / 2

        # Content loss : L1 distance
        content_loss = content_loss_coeff * tf.reduce_sum(tf.abs(sr - hr))

        # Perceptual loss
        hr_features = vgg_54()(hr)
        sr_features = vgg_54()(sr)
        perc_loss = tf.reduce_mean(tf.square(sr_features - hr_features))

        generator_loss = generator_loss + content_loss + perc_loss

        # Discriminator loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_output - tf.reduce_mean(sr_output),
                                                                             labels=tf.ones_like(hr_output))) / 2

        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sr_output - tf.reduce_mean(hr_output),
                                                                             labels=tf.zeros_like(sr_output))) / 2

        discriminator_loss = d_loss_real + d_loss_fake

        return generator_loss, discriminator_loss

# -----------------------------------------------------------
#  OPTIMIZER
# -----------------------------------------------------------

class Optimizer(object):
    @staticmethod
    def pretrain_optimizer(pretrain_learning_rate, pretrain_learning_rate_decay_step):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(pretrain_learning_rate, pretrain_learning_rate_decay_step, 0.5, staircase=True)
        pre_gen_optimizer = Adam(learning_rate=learning_rate)
        
        return pre_gen_optimizer
    
    @staticmethod
    def gan_optimizer(gan_learning_rate):
        boundaries = [50000, 100000, 200000, 300000]
        values = [gan_learning_rate, gan_learning_rate * 0.5, gan_learning_rate * 0.5 ** 2,
                  gan_learning_rate * 0.5 ** 3, gan_learning_rate * 0.5 ** 4]
        learning_rate = PiecewiseConstantDecay(boundaries, values)

        dis_optimizer = Adam(learning_rate=learning_rate)
        gen_optimizer = Adam(learning_rate=learning_rate)

        return dis_optimizer, gen_optimizer

# -----------------------------------------------------------
#  TRAINING GENERATOR
# -----------------------------------------------------------
class Pretrain():
    def __init__(self, generator):
        self.generator = generator 
        self.optimizer = Optimizer().pretrain_optimizer(pretrain_learning_rate=2e-4, pretrain_learning_rate_decay_step=20000)

    @tf.function
    def pretrain_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.generator(lr)
            pre_gen_loss = Loss().pretrain_loss(sr, hr)

        grad = tape.gradient(pre_gen_loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.generator.trainable_weights))
        return pre_gen_loss

    def train_gen(self, train_ds, epochs, print_every, save_every, log_filename, model_save_name):    
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_filename)), 'w+')
        log_file.close()

        epoch = 0
        for lr, hr in train_ds.take(epochs):
            epoch += 1    
            step_time = time.time()

            pre_gen_loss = self.pretrain_step(lr, hr)

            if epoch == 1 or epoch % print_every == 0:
                print("Epoch: [{}/{}], time: {:.3f}s, loss: {:3f} ".format(
                epoch, epochs, time.time() - step_time, pre_gen_loss))

                log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_filename)), 'a')
                log_file.write("Epoch: [{}/{}], time: {:.3f}s, loss: {:.3f} ".format(epoch, epochs, time.time() - step_time, pre_gen_loss))
                log_file.close()

            if epoch % save_every == 0:
                generator.save(model_save_dir + '/{}_{}.h5'.format(model_save_name, epoch))

# -----------------------------------------------------------
#  TRAINING GAN 
# -----------------------------------------------------------
class Train():
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
            
        self.generator_optimizer, self.discriminator_optimizer = Optimizer().gan_optimizer(gan_learning_rate=1e-4)

    @tf.function
    def train_step(self, lr, hr):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            # Generate fake hr images
            sr = self.generator(lr, training=True)

            # Training discriminator with two inputs hr and sr
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            generator_loss, discriminator_loss = Loss().train_loss(sr, hr, sr_output, hr_output) 
            
        # Adjusting gradients of generator
        gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))

        # Adjusting gradients of discriminator
        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

        return generator_loss, discriminator_loss 

    def train_gan(self, train_ds, epochs, print_every, save_every, log_filename, model_save_name):
        pls_metric = Mean()
        dls_metric = Mean()

        log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_filename)), 'w+')
        log_file.close()

        print('----- Start training -----')
        epoch = 0
        for lr, hr in train_ds.take(epochs):
            epoch += 1
            step_time = time.time()
            
            generator_loss, discriminator_loss = self.train_step(lr, hr)
            
            # Apply metrics
            pls_metric(generator_loss)
            dls_metric(discriminator_loss)
            
            # Update log every 100 epochs
            if epoch == 1 or epoch % print_every == 0:
                print('Epoch {}/{}, time: {:.3f}s, generator loss = {:.4f}, discriminator loss = {:.4f}'.format(epoch, epochs, time.time() - step_time, pls_metric.result(), dls_metric.result()))

                log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_filename)), 'a')
                log_file.write('Epoch {}/{}, time: {:.3f}s, generator loss = {:.4f}, discriminator loss = {:.4f}\n'.format(epoch, epochs, time.time() - step_time, pls_metric.result(), dls_metric.result()))
                log_file.close()

                pls_metric.reset_states()
                dls_metric.reset_states()

            # Save model every 500 epochs
            if epoch % save_every == 0:
                generator.save(model_save_dir + '/gen_{}_{}.h5'.format(model_save_name, epoch))
                discriminator.save(model_save_dir + '/dis_{}_{}.h5'.format(model_save_name, epoch))

# -----------------------------------------------------------
#  Command
# -----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--genpath", type=str, required=False)
    parser.add_argument("--dispath", type=str, required=False)

    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--print_every", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    
    parser.add_argument("--logname", type=str, required=True)
    parser.add_argument("--modelname", type=str, required=True)

    args = parser.parse_args()

    if args.type == "generator":
        print('START GENERATOR TRAINING SESSION')
        train_ds, num_train = DataLoader(dataset_name = 'div2k', subset='train').load_dataset()
        valid_ds, num_valid = DataLoader(dataset_name = 'div2k', subset='valid').load_dataset()
        generator = Generator().build(noise_shape=(None, None, 3))
        
        Pretrain(generator).train_gen(train_ds, 
                                        epochs = args.epochs, 
                                        print_every=args.print_every, 
                                        save_every=args.save_every, 
                                        log_filename=args.logname, 
                                        model_save_name=args.modelname)
        print('FINISHED.')
    
    if args.type == "gan":
        print('START GAN TRAINING SESSION')
        train_ds, val_ds, num_train, num_test = DataLoader(dataset_name=args.dataset).load_dataset()
        
        if args.genpath:
            generator = tf.keras.models.load_model(args.genpath)
        else:
            generator = Generator().build(noise_shape=(None, None, 3)) 

        if args.dispath:
            discriminator = tf.keras.models.load_model(args.dispath)
        else:
            discriminator = Discriminator.build(hr_shape=(128, 128, 3))
        
        Train(generator, discriminator).train_gan(train_ds, 
                                                    epochs = args.epochs, 
                                                    print_every=args.print_every, 
                                                    save_every=args.save_every, 
                                                    log_filename=args.logname, 
                                                    model_save_name=args.modelname)
        print('FINISHED.')