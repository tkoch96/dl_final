import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot
from data import Data
np.set_printoptions(threshold=np.nan)


#Fonts dataset assumed

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 64*64 # Number of pixels in image
NUM_CLASSES = 62; #all characters

#computer dependent paths
tom_path = 'fonts.hdf5'
sahil_path = '../../fonts.hdf5'

lib.print_model_settings(locals().copy())

def make_onehot(labs): #makes 1 hots from integers
    out_labs = []
    for lab in labs:
        out_labs.append(np.eye(NUM_CLASSES)[lab])
    return np.array(out_labs)

def make_image(labels_tensor, labels_np, z):
    np.savetxt('labels.csv',labels_np)
    return Generator(128, labels=labels_tensor,noise=z)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise, labels=None):
    if labels is not None:
        noise = tf.concat([noise,labels],1) #add the labels to the z vector for input to the generator
        output = lib.ops.linear.Linear('Generator.Input', SIZE_Z+NUM_CLASSES, 4*4*4*DIM, noise)
    else
        output = lib.ops.linear.Linear('Generator.Input',SIZE_Z,4*4*4*Dim,noise)
    output = tf.nn.relu(output)
    

    output = lib.ops.linear.Linear('Generator.1',4*4*4*DIM,8*8*4*DIM,output)
    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, 4*DIM, 8, 8])

    # C, H, W

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output) 
    output = tf.nn.relu(output)


    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output) 
    output = tf.nn.sigmoid(output)


    return tf.reshape(output, [-1, OUTPUT_DIM])

def recover_latent(inputs):
    INTER_DIM = 1024
    output = tf.reshape(inputs, [-1, 1, SIZE_IM, SIZE_IM])
  
    output = lib.ops.conv2d.Conv2D('Latent.1',1,DIM,5,output,stride=2) #DIM, 31, 31
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Latent.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D('Latent.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    
    output = lib.ops.conv2d.Conv2D('Latent.4', 4*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    output = lib.ops.linear.Linear('Latent.5', 4*4*4*DIM,INTER_DIM,output)
    output = LeakyReLU(output)
    output = lib.ops.linear.Linear('Latent.Output', INTER_DIM, SIZE_Z, output)
    return tf.reshape(output,[-1,SIZE_Z])

data_lab = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
z = tf.placeholder(tf.float32, shape=[BATCH_SIZE,SIZE_Z])
z_prime = recover_latent(Generator(BATCH_SIZE,noise=z))

latent_params = lib.params_with_name('Latent')

#latent is N(0,1)
mu = 0
sigma = 1

l2_cost = tf.reduce_mean(tf.pow(Generator(BATCH_SIZE,noise=z) - Generator(BATCH_SIZE,noise=z_prime)),2)
drift_cost = tf.reduce_mean(tf.abs((z_prime - mu)/sigma))
latent_cost = l2_cost + drift_cost


latent_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(latent_cost, var_list=latent_params)


gen_params = lib.params_with_name('Generator')
latent_params = lib.params_with_name('Latent')

saver1 = tf.train.Saver(gen_params)
saver2 = tf.train.Saver(latent_params)

# Train loop
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    saver1.restore(session, './fonts_checkpoints_r2/-37999') #get the latest generator weights to use
    for iteration in xrange(ITERS):
        input_lat = np.random.normal(mu,sigma,[BATCH_SIZE,SIZE_Z])
        _lat_cost, _ = session.run(
            [latent_cost, latent_train_op],
            feed_dict={z : input_lat}
        )
        if iteration % 1000 == 999:
            saver2.save(session, './fonts_checkpoints_latent/', global_step=iteration)
