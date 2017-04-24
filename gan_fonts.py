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

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
NUM_CLASSES = 62; #fonts
SIZE_Z = 512
SIZE_IM = 64
OUTPUT_DIM = SIZE_IM*SIZE_IM #size of image
TRAIN = 0
NUM_GEN = 100

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
    return Generator(SIZE_Z, labels=labels_tensor,noise=z)

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

def Generator(n_samples, noise=None, labels=None):
    if noise is None:
        noise = tf.random_normal([n_samples, SIZE_Z])

    noise = tf.concat([noise,labels],1) #add the labels to the z vector for input to the generator
    output = lib.ops.linear.Linear('Generator.Input', SIZE_Z+NUM_CLASSES, 4*4*4*DIM, noise)

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

def Discriminator(inputs,labels=None):
    INTER_DIM = 256

    lab_biases1 = lib.param('label_biases1',np.zeros(DIM, dtype='float32'))
    lab_wts1 = lib.param('label_weights1',np.float32(np.random.normal(size=[NUM_CLASSES,DIM])))

    lab_biases2 = lib.param('label_biases2',np.zeros(2 * DIM, dtype='float32'))
    lab_wts2 = lib.param('label_weights2',np.float32(np.random.normal(size=[DIM,2*DIM])))

    lab_biases3 = lib.param('label_biases3',np.zeros(4 * 4 * 4 * DIM, dtype='float32'))
    lab_wts3 = lib.param('label_weights3',np.float32(np.random.normal(size=[2 * DIM,4 * 4 * 4 * DIM])))


    #Bring the label information up to a sufficient size
    lab_out = tf.matmul(labels,lab_wts1) + lab_biases1
    lab_out = LeakyReLU(lab_out)
    lab_out = tf.matmul(lab_out,lab_wts2) + lab_biases2
    lab_out = LeakyReLU(lab_out)
    lab_out = tf.matmul(lab_out,lab_wts3) + lab_biases3
    lab_out = LeakyReLU(lab_out)


    output = tf.reshape(inputs, [-1, 1, SIZE_IM, SIZE_IM])
  
    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2) #DIM, 31, 31
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    
    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    #CONCAT WITH LABELS
    output = tf.concat([output, lab_out],1)
    output = lib.ops.linear.Linear('Discriminator.5', 4*4*4*2*DIM,INTER_DIM,output)
    output = LeakyReLU(output)
    output = lib.ops.linear.Linear('Discriminator.Output', INTER_DIM, 1, output)
    return tf.reshape(output, [-1])

real_data_ex = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data_lab = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
fake_data = Generator(BATCH_SIZE,labels=real_data_lab)

disc_real = Discriminator(real_data_ex,labels=real_data_lab)
disc_fake = Discriminator(fake_data,labels=real_data_lab)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

saver = tf.train.Saver(gen_params)



gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data_ex
interpolates = real_data_ex + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates,labels=real_data_lab), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5, 
    beta2=0.9
).minimize(disc_cost, var_list=disc_params)

clip_disc_weights = None



# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(SIZE_Z)).astype('float32'))
fixed_noise = tf.reshape(tf.tile(fixed_noise,tf.constant(NUM_CLASSES,shape=[1])), shape=[NUM_CLASSES,SIZE_Z])
rl = np.arange(0,NUM_CLASSES-1)

random_lab = tf.constant(make_onehot(rl).astype('float32'),shape=[NUM_CLASSES,NUM_CLASSES])
fixed_noise_samples = make_image(random_lab, rl, fixed_noise)



def generate_image(frame, spec_gen=None):
	if spec_gen is None: #training
	    samples = session.run(fixed_noise_samples)
	    lib.save_images.save_images(
	        samples.reshape((NUM_CLASSES, SIZE_IM, SIZE_IM)), 
	        './fonts_samples_r2/samples_{}.png'.format(frame)
	    )
	else: #already trained, generate pretty pixtors
		z = session.run(tf.constant(spec_gen[0]))
		l = session.run(tf.constant(spec_gen[1]))
		samples = session.run(Generator(SIZE_Z, labels=l,noise=z))
		lib.save_images.save_images(
			samples.reshape((NUM_CLASSES, SIZE_IM, SIZE_IM)),
			'./fonts_special_gen_r2/samples_{}.png'.format(frame)
		)

# Dataset iterator
#train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
dataset = Data(sahil_path, SIZE_Z, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield [images,make_onehot(targets)]
if TRAIN == 1: #train the model
	# Train loop
	with tf.Session() as session:

	    session.run(tf.initialize_all_variables())

	    for iteration in xrange(ITERS):
	        start_time = time.time()

	        if iteration > 0:
	            _data = dataset.serve_real()
	            _ = session.run(gen_train_op, feed_dict={real_data_lab : _data[1]})

	        
	        disc_iters = CRITIC_ITERS

	        for i in xrange(disc_iters):
	            _data = dataset.serve_real()
	            _disc_cost, _ = session.run(
	                [disc_cost, disc_train_op],
	                feed_dict={real_data_ex: _data[0], real_data_lab: _data[1]}
	            )
	            if clip_disc_weights is not None:
	                _ = session.run(clip_disc_weights)

	        lib.plot.plot('train disc cost', _disc_cost)
	        lib.plot.plot('time', time.time() - start_time)

	        # Calculate dev loss and generate samples every 100 iters
	        if iteration % 100 == 99:
	            dev_disc_costs = []
	            #for images,labs in dev_gen():

	            #    _dev_disc_cost = session.run(
	            #        disc_cost, 
	            #        feed_dict={real_data_ex: images, real_data_lab : make_onehot(labs)}
	            #    )
	            #    dev_disc_costs.append(_dev_disc_cost)
	            #lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
	            print iteration
	            generate_image(iteration)

	        if iteration % 1000 == 999:
	            saver.save(session, './fonts_checkpoints_r2/', global_step=iteration)
	        # Write logs every 100 iters
	        #if (iteration < 5) or (iteration % 100 == 99):
	        #    lib.plot.flush()

	        #lib.plot.tick()
else: #evaluate latent vectors
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		saver.restore(session, './fonts_checkpoints_r2/-37999')
		for i in range(NUM_GEN):
			latent = np.reshape(np.tile(np.random.normal(0,1,[SIZE_Z]),NUM_CLASSES),(NUM_CLASSES,SIZE_Z)).astype('float32')
			labels = make_onehot(np.arange(0,NUM_CLASSES)).astype('float32')
			generate_image(i, spec_gen=[latent,labels])