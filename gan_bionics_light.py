
# -*- coding: utf-8 -*-


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
import load_image
import load_image_eigen


lib.print_model_settings(locals().copy())

def parse_args():
    parser = argparse.ArgumentParser(description='cut images')
    parser.add_argument('--MODE', dest='MODE', help='dcgan, wgan, or wgan-gp', default='wgan-gp', type=str)
    parser.add_argument('--output_path', dest='output_path', help='the output path', default='e:/project/project/image/input_3_64_10000_rot/10', type=str)
    parser.add_argument('--DATA_DIR', dest='DATA_DIR', help='the input path', default='e:/project/project/image/input_3_64_10000_rot/10', type=str)
    parser.add_argument('--color_mode', dest='color_mode', help='rgb or gray-scale', default='rgb', type=str)
    parser.add_argument('--DIM', dest='DIM', help='Model dimensionality',type=int, default=64)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=128)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=200000)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=64*64*3)
    parser.add_argument('--output_lenth', dest='output_lenth', help='lenth of the output images',type=int, default=64)
    parser.add_argument('--img_num', dest='img_num', help='the number of the output images', type=int, default=4096)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    def LeakyReLU(x, n, alpha=0.2):
        return tf.maximum(alpha*x, x,name=n)
    
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
    
    def Generator(n_samples, noise=None, dim=args.DIM,  nonlinearity=tf.nn.relu,reuse=False):
  
    
        if noise is None:
            noise = tf.random_uniform([n_samples, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
        
        if dim==64:
            with tf.variable_scope('gen') as scope:
                if reuse:
                    scope.reuse_variables()
                w1 = tf.get_variable('w1', shape=[128, 4*4*8*dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
                b1 = tf.get_variable('b1', shape=[4*4*8*dim], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
                flat_conv1 = tf.add(tf.matmul(noise, w1), b1, name='flat_conv1')
                # 4*4*512
                conv1 = tf.reshape(flat_conv1, shape=[-1, 4, 4, 8*dim], name='conv1')
                act1 = tf.nn.relu(conv1, name='act1')
                # 8*8*256
                conv2 = tf.layers.conv2d_transpose(act1, 4*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv2')
                act2 = tf.nn.relu(conv2, name='act2')
                # 16*16*128
                conv3 = tf.layers.conv2d_transpose(act2, 2*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv3')
                act3 = tf.nn.relu(conv3, name='act3')
                # 32*32*64
                conv4 = tf.layers.conv2d_transpose(act3, dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv4')
                act4 = tf.nn.relu(conv4, name='act4')
                # 64*64*3
                conv5 = tf.layers.conv2d_transpose(act4, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv5')
                act5 = tf.nn.tanh(conv5, name='act5')
            
                return tf.reshape(act5, [-1, args.OUTPUT_DIM])
                
        if dim==128:
            with tf.variable_scope('gen') as scope:
                if reuse:
                    scope.reuse_variables()
                w1 = tf.get_variable('w1', shape=[128, 4*4*16*dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
                b1 = tf.get_variable('b1', shape=[4*4*16*dim], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
                flat_conv1 = tf.add(tf.matmul(noise, w1), b1, name='flat_conv1')
                # 4*4*2048
                conv1 = tf.reshape(flat_conv1, shape=[-1, 4, 4, 16*dim], name='conv1')
                act1 = tf.nn.relu(conv1, name='act1')
                # 8*8*1024
                conv2 = tf.layers.conv2d_transpose(act1, 8*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv2')
                act2 = tf.nn.relu(conv2, name='act2')
                # 16*16*512
                conv3 = tf.layers.conv2d_transpose(act2, 4*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv3')
                act3 = tf.nn.relu(conv3, name='act3')
                # 32*32*256
                conv4 = tf.layers.conv2d_transpose(act3, 2*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv4')
                act4 = tf.nn.relu(conv4, name='act4')
                # 64*64*128
                conv5 = tf.layers.conv2d_transpose(act4, dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv5')
                act5 = tf.nn.tanh(conv5, name='act5')
                #128*128*3
                conv6 = tf.layers.conv2d_transpose(act5, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   name='conv6')
                act6 = tf.nn.relu(conv6, name='act6')                
            
                return tf.reshape(act6, [-1, args.OUTPUT_DIM])            
            

    
    def Discriminator(inputs, dim=args.DIM, nonlinearity=LeakyReLU,reuse=False):
    

        if dim==64:

            with tf.variable_scope('dis') as scope:
                if reuse:
                    scope.reuse_variables()
                # 64643
                inputs = tf.reshape(inputs, [-1, args.DIM, args.DIM, 3])    
                # 323264
                conv1 = tf.layers.conv2d(inputs, dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv1')
                act1 = LeakyReLU(conv1, n='act1')
                # 1616128
                conv2 = tf.layers.conv2d(act1, 2*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv2')
                act2 = LeakyReLU(conv2, n='act2')
                # 88256
                conv3 = tf.layers.conv2d(act2, 4*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv3')
                act3 = LeakyReLU(conv3, n='act3')
                #44512
                conv4 = tf.layers.conv2d(act3, 8*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv4')
                act4 = LeakyReLU(conv4, n='act4')                
        

                fc1 = tf.reshape(act4, shape=[-1, 4*4*8*dim], name='fc1')
                w1 = tf.get_variable('w1', shape=[fc1.shape[1], 1], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
                b1 = tf.get_variable('b1', shape=[1], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        
                # wgan just get rid of the sigmoid
                output = tf.add(tf.matmul(fc1, w1), b1, name='output')
                return tf.reshape(output, [-1])           

        if dim==128:
            with tf.variable_scope('dis') as scope:
                if reuse:
                    scope.reuse_variables()
                # 128*128*3
                inputs = tf.reshape(inputs, [-1,args.DIM, args.DIM,3])    
                # 64*64*128
                conv1 = tf.layers.conv2d(inputs, dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv1')
                act1 = LeakyReLU(conv1, n='act1')
                # 32*32*256
                conv2 = tf.layers.conv2d(act1, 2*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv2')
                act2 = LeakyReLU(conv2, n='act2')
                # 16*16*512
                conv3 = tf.layers.conv2d(act2, 4*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv3')
                act3 = LeakyReLU(conv3, n='act3')
                # 8*8*1024
                conv4 = tf.layers.conv2d(act3, 8*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv4')
                act4 = LeakyReLU(conv4, n='act4') 
                # 4*4*2048
                conv5 = tf.layers.conv2d(act4, 16*dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         name='conv5')
                act5 = LeakyReLU(conv5, n='act5')                
        

                fc1 = tf.reshape(act5, shape=[-1, 4*4*16*dim], name='fc1')
                w1 = tf.get_variable('w1', shape=[fc1.shape[1], 1], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
                b1 = tf.get_variable('b1', shape=[1], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
        
                # wgan just get rid of the sigmoid
                output = tf.add(tf.matmul(fc1, w1), b1, name='output')
                return tf.reshape(output, [-1])                  
            

        

                


    # Train loop
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, args.DIM, args.DIM])
        real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [args.BATCH_SIZE, args.OUTPUT_DIM])
        eigen = tf.placeholder(tf.float32, shape=[args.BATCH_SIZE,1])
        
        fake_data = Generator(args.BATCH_SIZE)
        
        disc_real = Discriminator(real_data)
        disc_fake = Discriminator(fake_data,reuse=True)
        
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
            gradients = tf.gradients(Discriminator(interpolates,reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += args.LAMBDA*gradient_penalty
        
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'dis' in var.name]
            g_vars = [var for var in t_vars if 'gen' in var.name]
            disc_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=d_vars)
            gen_train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=g_vars)
    
        
            clip_disc_weights = None            


        

        
    
        
        
        # For saving samples
        fixed_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples = Generator(args.BATCH_SIZE, noise=fixed_noise,reuse=True)          
        def generate_image(iteration):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples.reshape((args.BATCH_SIZE, 3, args.DIM, args.DIM)), args.model_dir +'/samples_{}.png'.format(iteration))
        
        # Dataset iterator

        train_gen = load_image_eigen.load(args.BATCH_SIZE, data_dir=args.DATA_DIR, dim=args.DIM,num=args.img_num,count=args.restore_index)
        def inf_train_gen():
            while True:
                for (images,eigenvalues) in train_gen():
                    yield images,eigenvalues  
        
        # Save a batch of ground-truth samples
        _x = inf_train_gen().__next__()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:args.BATCH_SIZE]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((args.BATCH_SIZE, 3, 64, 64)), 'samples_groundtruth.png')
        
        
        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        
        saver = tf.train.Saver(max_to_keep=40)
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = args.restore_index + 1


        from functools import reduce
        from operator import mul
        
        def get_num_params():
            num_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                num_params += reduce(mul, [dim.value for dim in shape], 1)
            return num_params  
        print('haha')
        print(get_num_params())

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
                _data,_eigen = gen.__next__()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data_conv:_data, eigen:_eigen}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
            
            print("iter: %d   disc_cost: %f"%(index,_disc_cost))
            # Calculate dev loss and generate samples every 100 iters
            if index % 10 == 9:
                generate_image(index)
                saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                #saver.save(session, 'wgangp_bionics' + '.cptk')
            # Write logs every 100 iters
            if (index < 5) or (index % 10 == 9):
                lib.plot.flush()

    
            lib.plot.tick()
            index = index + 1
