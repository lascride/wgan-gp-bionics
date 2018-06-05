# -*- coding: utf-8 -*-
###############################################################################
#Step 1: Learning Approximate Manifold
###############################################################################

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
    parser = argparse.ArgumentParser(description='Step 1: Learning Approximate Manifold')
    
    #Common settings
    parser.add_argument('--DATA_DIR', dest='DATA_DIR', help='the input path', default='e:/project/project/image/input_bamboo_64', type=str)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--img_num', dest='img_num', help='the number of images in dataset', type=int, default=24576)
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)

    #WGAN settings
    parser.add_argument('--DIM', dest='DIM', help='Model dimensionality/image scale',type=int, default=64)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=128)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=50000)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels',type=int, default=64*64*3)

    #Jacobian settings
    parser.add_argument('--jacobian_num', dest='jacobian_num', help='the subdimension to calcualte jacobians', type=int, default=None)
    parser.add_argument('--delta', dest='delta', help='the step size to calcualte jacobians', type=float, default=0.0001)

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
    
    
    def LeakyReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name+'.Linear', 
            n_in, 
            n_out, 
            inputs,
            initialization='he'
        )
        return LeakyReLU(output)
    
    def Generator(n_samples, noise=None, dim=args.DIM, bn=True, nonlinearity=tf.nn.relu):
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
    
        if noise is None:
            #noise = tf.random_normal([n_samples, 128])
            noise = tf.random_uniform([n_samples, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
    
        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])
        #512*4*4
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
        #256*8*8
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
        #128*16*16
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
        #64*32*32
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        #3*64*64

        output = tf.tanh(output)
    
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return tf.reshape(output, [-1, args.OUTPUT_DIM])
    
    
    def Discriminator(inputs, dim=args.DIM, bn=True, nonlinearity=LeakyReLU):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
    
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
        #64*32*32
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
        #128*16*16
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        #256*8*8
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        #512*4*4
        output = nonlinearity(output)
    
        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)
        #scalar
        
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return tf.reshape(output, [-1])    
    

    
    # Train loop
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        
        #rescale color space into [-1,1]
        real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, 64, 64])
        real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [args.BATCH_SIZE, args.OUTPUT_DIM])
       
        g_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
        fake_data = Generator(args.BATCH_SIZE,noise = g_noise)
        
        disc_real = Discriminator(real_data)
        disc_fake = Discriminator(fake_data)
        
        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')
        
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        
        
        
        if args.jacobian_num:
            #Whether to include Jacobian loss

            g_noise_tile = tf.tile(g_noise[:,np.newaxis,:],[1,args.jacobian_num,1])
            g_noise_tile = tf.reshape(g_noise_tile, [args.BATCH_SIZE*args.jacobian_num,128])
            
            diagonal=[]
            for i in range(args.jacobian_num):
                diagonal.append(1)
                
            eye_label = tf.diag(diagonal) #identity matrix
            eye_label = tf.cast(tf.tile(eye_label[np.newaxis,:], [args.BATCH_SIZE, 1, 1]),tf.float32)
        
            diagonal=[]
            for i in range(128):
                diagonal.append(1)    
            #Randomly choose 8 coordinates to calculate jacobian
            eye_nz = tf.random_shuffle(tf.diag(diagonal)[0:args.jacobian_num])
            eye_nz = tf.cast(tf.tile(eye_nz[np.newaxis,:],[args.BATCH_SIZE, 1, 1]),tf.float32)
            pos_noise = tf.reshape(args.delta*eye_nz,[args.BATCH_SIZE*args.jacobian_num,128])
            
            Jx = (Generator(args.BATCH_SIZE*args.jacobian_num,noise=g_noise_tile + pos_noise)-
                    Generator(args.BATCH_SIZE*args.jacobian_num,noise=g_noise_tile - pos_noise))/(2*args.delta)
            Jx = tf.reshape(Jx,[args.BATCH_SIZE,args.jacobian_num,-1])
            Jx_T = tf.transpose(Jx,[0,2,1])
            
            errOrth = tf.reduce_mean(tf.abs(tf.matmul(Jx,Jx_T)-eye_label))
            
            gen_cost = gen_cost + errOrth*0.01            
            
        
        
        
        
        #Gradient penalty of Generator in WGAN
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
            learning_rate=2e-3, 
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-3, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)
        

        
  
        
        
        # For saving samples
        fixed_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples = Generator(args.BATCH_SIZE, noise=fixed_noise)          
        def generate_image(iteration):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples.reshape((args.BATCH_SIZE, 3, 64, 64)), args.model_dir +'/samples_{}.png'.format(iteration))
        
        
        
        
        # Dataset iterator
        train_gen = load_image.load(args.BATCH_SIZE, data_dir=args.DATA_DIR, num=args.img_num)
        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images
 

       
        # Save a batch of ground-truth samples
        _x = inf_train_gen().__next__()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:args.BATCH_SIZE]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((args.BATCH_SIZE, 3, 64, 64)), 'samples_groundtruth.png')



        
        saver = tf.train.Saver(max_to_keep=40)
        session.run(tf.initialize_all_variables())
        
        #restore the model
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1        

            

        gen = inf_train_gen()   
        for iteration in range(args.ITERS):
            start_time = time.time()
            
            
            if iteration > 0:
                _ = session.run(gen_train_op)
    

            disc_iters = args.CRITIC_ITERS
            
            for i in range(disc_iters):
                _data = gen.__next__()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data_conv:_data}
                )

    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
  
          
            print("iter: %d   disc_cost: %f"%(index,_disc_cost))
            
            # Calculate loss and generate samples
            if index % 100 == 99:
                if args.jacobian_num:
                    _gen_cost, _errOrth = session.run([gen_cost, errOrth],feed_dict={real_data_conv:_data})
                    print('_gen_cost')
                    print(_gen_cost)
                    print('_errOrth')
                    print(_errOrth)

                saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                                
            if (index < 5) or (index % 20 == 19):
                generate_image(index)
                lib.plot.flush()

    
            lib.plot.tick()
            index = index + 1
