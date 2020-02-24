import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean
import tensorflow.keras.backend as K
from tensorflow.python.data.experimental import AUTOTUNE

from tqdm import tqdm

import model, data

# -----------------------------------------------------------
#  LOSSES
# -----------------------------------------------------------
class Loss(object):
    def pretrain_loss(self, sr, hr):
        pre_gen_loss = tf.reduce_mean(tf.reduce_mean(tf.square(sr - hr), axis=3))
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
    """class to build optimizers"""
        
    def pretrain_optimizer(self):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(pretrain_lr, pretrain_lr_decay_step, 0.5, staircase=True)
        pre_gen_optimizer = Adam(learning_rate=learning_rate)
        
        return pre_gen_optimizer

    def gan_optimizer(self):
        boundaries = [50000, 100000, 200000, 300000]
        values = [gan_lr, gan_lr * 0.5, gan_lr * 0.5 ** 2,
                  gan_lr * 0.5 ** 3, gan_lr * 0.5 ** 4]
        learning_rate = PiecewiseConstantDecay(boundaries, values)

        dis_optimizer = Adam(learning_rate=learning_rate)
        gen_optimizer = Adam(learning_rate=learning_rate)

        return dis_optimizer, gen_optimizer

# -----------------------------------------------------------
#  TRAINING GENERATOR
# -----------------------------------------------------------

@tf.function
def pretrain_step(lr, hr):
    with tf.GradientTape() as tape:
        sr = pre_gen(lr)
        pre_gen_loss = Loss().pretrain_loss(sr, hr)

    grad = tape.gradient(pre_gen_loss, pre_gen.trainable_weights)
    pre_gen_optimizer.apply_gradients(zip(grad, pre_gen.trainable_weights))
    return pre_gen_loss

def train_gen(train_ds, epochs = 100, log_name="pretrain-loss"):    
    log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_name)), 'w+')
    log_file.close()
    
    epoch = 0
    for lr, hr in train_ds.take(epochs):
        epoch += 1    
        step_time = time.time()
        
        pre_gen_loss = pretrain_step(lr, hr)
        
        if epoch == 1 or epoch % 50 == 0:
            print("Epoch: [{}/{}], time: {:.3f}s, loss: {:3f} ".format(
            epoch, epochs, time.time() - step_time, pre_gen_loss))
            
            log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_name)), 'a')
            log_file.write("Epoch: [{}/{}], time: {:.3f}s, loss: {:.3f} ".format(epoch, epochs, time.time() - step_time, pre_gen_loss))
            log_file.close()
        
        if epoch % 100 == 0:
            pre_gen.save(model_save_dir + '/pretrain_genx4_2202%d.h5' % epoch)

# -----------------------------------------------------------
#  TRAINING GAN 
# -----------------------------------------------------------
@tf.function
def train_step(lr, hr):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        # Generate fake hr images
        sr = generator(lr, training=True)

        # Training discriminator with two inputs hr and sr
        hr_output = discriminator(hr, training=True)
        sr_output = discriminator(sr, training=True)

        generator_loss, discriminator_loss = Loss().train_loss(lr, hr, sr)
        
    # Adjusting gradients of generator
    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))

    # Adjusting gradients of discriminator
    gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

    return generator_loss, discriminator_loss 

def train_gans(train_ds, epochs=200000, log_name="gan-loss"):
    pls_metric = Mean()
    dls_metric = Mean()

    log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_name)), 'w+')
    log_file.close()

    print('----- Start training -----')
    epoch = 0
    for lr, hr in train_ds.take(epochs):
        epoch += 1
        step_time = time.time()
        
        generator_loss, discriminator_loss = train_step(lr, hr)
        
        # Apply metrics
        pls_metric(generator_loss)
        dls_metric(discriminator_loss)
        
        # Update log every 100 epochs
        if epoch == 1 or epoch % 100 == 0:
            print('Epoch {}/{}, time: {:.3f}s, generator loss = {:.4f}, discriminator loss = {:.4f}'.format(epoch, epochs, time.time() - step_time, pls_metric.result(), dls_metric.result()))

            log_file = open(os.path.join(LOG_DIR, '{}.txt'.format(log_name)), 'a')
            log_file.write('Epoch {}/{}, time: {:.3f}s, generator loss = {:.4f}, discriminator loss = {:.4f}\n'.format(epoch, epochs, time.time() - step_time, pls_metric.result(), dls_metric.result()))
            log_file.close()

            pls_metric.reset_states()
            dls_metric.reset_states()

        # Save model every 500 epochs
        if epoch % 500 == 0:
            generator.save(model_save_dir + '/gen_model_{}.h5'.format(epoch))
            discriminator.save(model_save_dir + '/dis_model_{}.h5'.format(epoch))

# -----------------------------------------------------------
#  Command
# -----------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--epochs", type=str, required=True)
    parser.add_argument("--logname", type=str, required=False)

    args = parser.parse_args()

    if args.train:
        if args.type == "generator":
            print('START GENERATOR TRAINING SESSION')
            train_ds, val_ds, num_train, num_test = DataLoader().load_dataset()
            generator = Generator().build()
            pretrain_optimizer = Optimizer().pretrain_optimizer()
            train_gen(train_ds, epochs = args.epochs, log_name=args.logname)
            print('FINISHED.')
        
        if args.type == "gan":
            print('START GAN TRAINING SESSION')
            train_ds, val_ds, num_train, num_test = DataLoader().load_dataset()
            generator = Generator().build()
            discriminator = Discriminator.build()
            generator_optimizer, discriminator_optimizer = Optimizer().gan_optimizer()
            train_gans(train_ds, epochs = args.epochs, log_name=args.logname)
            print('FINISHED.')