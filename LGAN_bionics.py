# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:32:20 2018

@author: lenovo
"""

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
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models_lgan',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    
    parser.add_argument('--jacobian_num', dest='jacobian_num', help='the subdimension to calcualte jacobians', type=int, default=8)
    parser.add_argument('--delta', dest='delta', help='the step size to calcualte jacobians', type=float, default=0.0001)

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
    
    def Generator(inputs,n_samples, noise=None, dim=args.DIM,  nonlinearity=tf.nn.relu,reuse=False, is_train = True):
  
    
        if noise is None:
            noise = tf.random_normal([n_samples, 128], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            
        noise = noise[:,np.newaxis,np.newaxis,:]

        with tf.variable_scope('gen') as scope:

            if reuse:
                scope.reuse_variables()
            #64*64*3
            inputs = tf.reshape(inputs, [-1, args.DIM, args.DIM, 3])  
            # 32*32*64
            conv1 = tf.layers.conv2d(inputs, dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv1')
            act1 = LeakyReLU(conv1, n='act1')
            # 16*16*128
            conv2 = tf.layers.conv2d(act1, 2*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
            act2 = LeakyReLU(bn2, n='act2')
            # 8*8*256
            conv3 = tf.layers.conv2d(act2, 4*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv3')
            bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
            act3 = LeakyReLU(bn3, n='act3')
            # 4*4*512
            conv4 = tf.layers.conv2d(act3, 8*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv4')
            bn4 = tf.layers.batch_normalization(conv4, training=is_train, name='bn4')
            act4 = LeakyReLU(bn4, n='act4') 
            
            # 1*1*128
            conv5 = tf.layers.conv2d(act4, 128, kernel_size=[4, 4], strides=[1, 1], padding="valid",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv5')
            bn5 = tf.layers.batch_normalization(conv5, training=is_train, name='bn5')
            
            
                           
            local = bn5 + noise
            act5 = LeakyReLU(local, n='act5')
            # 4*4*512
            deconv1 = tf.layers.conv2d_transpose(act5, 8*dim, kernel_size=[4, 4], strides=[1, 1], padding="valid",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='deconv1')
            debn1 = tf.layers.batch_normalization(deconv1, training=is_train, name='debn1')
            deact1 = LeakyReLU(debn1, n='deact1')               
            print("haha")
            print(tf.shape(deact1))
                
            # # 8*8*256
            deconv2 = tf.layers.conv2d_transpose(deact1, 4*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='deconv2')
            debn2 = tf.layers.batch_normalization(deconv2, training=is_train, name='debn2')
            deact2 = tf.nn.relu(debn2, name='deact2')
            # 16*16*128
            deconv3 = tf.layers.conv2d_transpose(deact2, 2*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='deconv3')
            debn3 = tf.layers.batch_normalization(deconv3, training=is_train, name='debn3')
            deact3 = tf.nn.relu(debn3, name='deact3')
            # 32*32*64
            deconv4 = tf.layers.conv2d_transpose(deact3, dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='deconv4')
            debn4 = tf.layers.batch_normalization(deconv4, training=is_train, name='debn4')
            deact4 = tf.nn.relu(debn4, name='deact4')
            # 64*64*3
            deconv5 = tf.layers.conv2d_transpose(deact4, 3, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               name='deconv5')
            deact5 = tf.nn.tanh(deconv5, name='deact5')
    
            return tf.reshape(deact5, [-1, args.OUTPUT_DIM])
            
         
            

    
    def Discriminator(inputs, dim=args.DIM, nonlinearity=LeakyReLU,reuse=False,is_train = True):
  
    
        with tf.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()
            # 16*16*32
            inputs = tf.reshape(inputs, [-1, args.DIM, args.DIM, 3])    
            conv1 = tf.layers.conv2d(inputs, dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv1')
            act1 = LeakyReLU(conv1, n='act1')
            # 8*8*64
            conv2 = tf.layers.conv2d(act1, 2*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv2')
            bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
            act2 = LeakyReLU(bn2, n='act2')
            # 4*4*128
            conv3 = tf.layers.conv2d(act2, 4*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv3')
            bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
            act3 = LeakyReLU(bn3, n='act3')
            
            conv4 = tf.layers.conv2d(act3, 8*dim, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     name='conv4')
            act4 = LeakyReLU(conv4, n='act4') 
            bn4 = tf.layers.batch_normalization(act4, training=is_train, name='bn4')
    

            fc1 = tf.reshape(bn4, shape=[-1, 4*4*8*dim], name='fc1')
            w1 = tf.get_variable('w1', shape=[fc1.shape[1], 1], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b1', shape=[1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
    
            # wgan just get rid of the sigmoid
            output = tf.add(tf.matmul(fc1, w1), b1, name='output')
            return tf.reshape(output, [-1])   
                
                

        
            
    real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, args.DIM, args.DIM])
    real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [args.BATCH_SIZE, args.OUTPUT_DIM])
    
    fake_data = Generator(real_data,args.BATCH_SIZE)
    fake_data_0 = Generator(real_data,args.BATCH_SIZE, noise = tf.constant(0.0, shape=[args.BATCH_SIZE, 128]),reuse=True )
    
    
    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data,reuse=True)
    disc_fake_0 = Discriminator(fake_data_0,reuse=True)
    
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')
    

    

    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = 0.1*tf.reduce_mean(disc_fake_0) + 0.9*tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    disc_cost_0 = tf.reduce_mean(disc_fake_0) - tf.reduce_mean(disc_real)
    
    errL1 = tf.reduce_mean(tf.abs(fake_data_0-real_data))
    
    ##Jacobian
    real_data_tile = tf.tile(real_data[np.newaxis,:],[args.jacobian_num,1,1])
    real_data_tile = tf.transpose(real_data_tile,[1,0,2])
    real_data_tile = tf.reshape(real_data_tile, [args.BATCH_SIZE*args.jacobian_num,12288])
    
    
    
    diagonal=[]
    for i in range(args.jacobian_num):
        diagonal.append(1)
        
    eye_label = tf.diag(diagonal)
    eye_label = tf.cast(tf.tile(eye_label[np.newaxis,:], [args.BATCH_SIZE, 1, 1]),tf.float32)

    diagonal=[]
    for i in range(128):
        diagonal.append(1)    
    eye_nz = tf.random_shuffle(tf.diag(diagonal)[0:args.jacobian_num])
    eye_nz = tf.cast(tf.tile(eye_nz[np.newaxis,:],[args.BATCH_SIZE, 1, 1]),tf.float32)
    pos_noise = tf.reshape(args.delta*eye_nz,[args.BATCH_SIZE*args.jacobian_num,128])
    
    Jx = (Generator(real_data_tile,args.BATCH_SIZE*args.jacobian_num,noise=pos_noise,reuse=True)-
            Generator(real_data_tile,args.BATCH_SIZE*args.jacobian_num,noise=-pos_noise,reuse=True))/(2*args.delta)
    Jx = tf.reshape(Jx,[args.BATCH_SIZE,args.jacobian_num,-1])
    Jx_T = tf.transpose(Jx,[0,2,1])
    
    errOrth = tf.reduce_mean(tf.abs(tf.matmul(Jx,Jx_T)-eye_label))
    
    
    gen_cost = gen_cost + errL1*20 + errOrth*0.01
    
    


    
    

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
    disc_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=d_vars)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=g_vars)
load

    clip_disc_weights = None  
    
        
    train_gen = load_image.load(args.BATCH_SIZE, data_dir=args.DATA_DIR, dim=args.DIM,num=args.img_num)
    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images  
                

    gen = inf_train_gen()
    # Train loop
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 

    with tf.Session(config=config) as session:
        
        


        

        
    
        
        
        # For saving samples
        fixed_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        def generate_image(real_data,iteration):
            noises = session.run(fixed_noise)
            samples = Generator(real_data,args.BATCH_SIZE, noise=noises,reuse=True)          

            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples.reshape((args.BATCH_SIZE, 3, args.DIM, args.DIM)), args.model_dir +'/samples_{}.png'.format(iteration))
        
        # Dataset iterator


        

        
        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        
        saver = tf.train.Saver(max_to_keep=40)
        session.run(tf.initialize_all_variables())
        index = 1
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
                _real_data,_disc_cost, _ = session.run(
                    [real_data,disc_cost, disc_train_op],
                    feed_dict={real_data_conv:_data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)
    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
            
            print("iter: %d   disc_cost: %f"%(index,_disc_cost))
            # Calculate dev loss and generate samples every 100 iters
            if index % 10 == 9:
                generate_image(_real_data,index)
                saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                #saver.save(session, 'wgangp_bionics' + '.cptk')
            # Write logs every 100 iters
            if (index < 5) or (index % 10 == 9):
                lib.plot.flush()

    
            lib.plot.tick()
            index = index + 1
