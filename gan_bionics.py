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
import argparse


lib.print_model_settings(locals().copy())

def parse_args():
    parser = argparse.ArgumentParser(description='cut images')
    parser.add_argument('--MODE', dest='MODE', help='dcgan, wgan, or wgan-gp', default='wgan-gp', type=str)
    parser.add_argument('--output_path', dest='output_path', help='the output path', default='e:/project/project/image/input_3_64_10000_rot', type=str)
    parser.add_argument('--input_path', dest='input_path', help='the input path', default='e:/project/project/image/input_3_64_10000_rot', type=str)
    parser.add_argument('--color_mode', dest='color_mode', help='rgb or gray-scale', default='rgb', type=str)
    parser.add_argument('--DIM', dest='DIM', help='Model dimensionality',type=int, default=64)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=50)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=200000)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=784)
    parser.add_argument('--output_lenth', dest='output_lenth', help='lenth of the output images',type=int, default=64)
    parser.add_argument('--img_num', dest='img_num', help='the number of the output images', type=int, default=4096)
    parser.add_argument('--model_dir', type=str, default='models',
                        help='directory to save models')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
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
    
    def Generator(n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])
    
        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*args.DIM, noise)
        if args.MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*args.DIM, 4, 4])
    
        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*args.DIM, 2*args.DIM, 5, output)
        if args.MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)
    
        output = output[:,:,:7,:7]
    
        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*args.DIM, args.DIM, 5, output)
        if args.MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.5', args.DIM, 1, 5, output)
        output = tf.nn.sigmoid(output)
    
        return tf.reshape(output, [-1, args.OUTPUT_DIM])
    
    def Discriminator(inputs):
        output = tf.reshape(inputs, [-1, 1, 28, 28])
    
        output = lib.ops.conv2d.Conv2D('Discriminator.1',1,args.DIM,5,output,stride=2)
        output = LeakyReLU(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.2', args.DIM, 2*args.DIM, 5, output, stride=2)
        if args.MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = LeakyReLU(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*args.DIM, 4*args.DIM, 5, output, stride=2)
        if args.MODE == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = LeakyReLU(output)
    
        output = tf.reshape(output, [-1, 4*4*4*args.DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*args.DIM, 1, output)
    
        return tf.reshape(output, [-1])
    
    real_data = tf.placeholder(tf.float32, shape=[args.BATCH_SIZE, args.OUTPUT_DIM])
    fake_data = Generator(args.BATCH_SIZE)
    
    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)
    
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    
    if args.MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    
        gen_train_op = tf.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(disc_cost, var_list=disc_params)
    
        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var, 
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)
    
    elif args.MODE == 'wgan-gp':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    
        alpha = tf.random_uniform(
            shape=[args.BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += args.LAMBDA*gradient_penalty
    
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
    
    elif args.MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_fake, 
            tf.ones_like(disc_fake)
        ))
    
        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_fake, 
            tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_real, 
            tf.ones_like(disc_real)
        ))
        disc_cost /= 2.
    
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(disc_cost, var_list=disc_params)
    
        clip_disc_weights = None
    
    # For saving samples
    
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        lib.save_images.save_images(
            samples.reshape((128, 28, 28)), 
            args.model_dir +'/samples_{}.png'.format(frame)
        )
    
    # Dataset iterator
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images
    
    
    # For saving samples
    fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples = Generator(128, noise=fixed_noise)
    
    train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)
    # Train loop
    saver = tf.train.Saver()
    with tf.Session() as session:
    
        session.run(tf.initialize_all_variables())
    
        gen = inf_train_gen()

 
        for iteration in range(args.ITERS):
            start_time = time.time()
    
            if iteration > 0:
                _ = session.run(gen_train_op)
    
            if args.MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = args.CRITIC_ITERS
            for i in range(disc_iters):
                _data = gen.__next__()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
    
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
    
                generate_image(iteration, _data)
                saver.save(session, args.model_dir + '/wgangp_' + str(iteration) + '.cptk')
            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()
    
            lib.plot.tick()
