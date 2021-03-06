# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:46:32 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 04:33:16 2018

@author: lenovo
"""
import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)  

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
    parser.add_argument('--output_path', dest='output_path', help='the output path', default='e:/project/project/image/input_bamboo_64', type=str)
    parser.add_argument('--DATA_DIR', dest='DATA_DIR', help='the input path', default='e:/project/project/image/input_bamboo_64', type=str)
    parser.add_argument('--color_mode', dest='color_mode', help='rgb or gray-scale', default='rgb', type=str)
    parser.add_argument('--DIM', dest='DIM', help='Model dimensionality',type=int, default=64)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=128)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=200000)
    parser.add_argument('--EIGEN_ITERS', dest='EIGEN_ITERS', help='How many iterations to train Eigener',type=int, default=3000)
   
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=64*64*3)
    parser.add_argument('--output_lenth', dest='output_lenth', help='lenth of the output images',type=int, default=64)
    parser.add_argument('--img_num', dest='img_num', help='the number of the output images', type=int, default=4096)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    parser.add_argument('--infer', dest='infer', help='infer', default='False', type=str)

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
    
    def Generator(n_samples, noise=None, dim=args.DIM, bn=True, nonlinearity=tf.nn.relu):
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
    
        if noise is None:
            #noise = tf.random_normal([n_samples, 128])
            noise = tf.random_uniform([n_samples, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
    
        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    
        output = nonlinearity(output)
    
        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        
        output_image = tf.tanh(output)#128*3*64*64
        
  
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return output_image


    def Eigener(inputs, dim=args.DIM, bn=True, nonlinearity=LeakyReLU):
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
     
        
        output = lib.ops.conv2d.Conv2D('Eigener.1', 3, dim, 5, inputs, stride=2)
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Eigener.2', dim, 2*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Eigener.3', 2*dim, 4*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Eigener.4', 4*dim, 8*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Eigener.Output', 4*4*8*dim, 1, output) 
        
        eigen = tf.tanh(output)#128*1
        

        
    
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return eigen

    
    
    def Discriminator(inputs, dim=args.DIM, bn=True, nonlinearity=LeakyReLU):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
    
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    
        output = nonlinearity(output)
    
        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)
    
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return tf.reshape(output, [-1])    
    

    
    # Train loop
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        

        real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, 64, 64])
        real_data = 2*((tf.cast(real_data_conv, tf.float32)/255.)-.5)
        
        
        
        true_eigen = tf.placeholder(tf.float32, shape=[args.BATCH_SIZE,1])
        true_eigen_tile = tf.tile(true_eigen[:,:,np.newaxis,np.newaxis],[1,3,64,64])
        
        
        
        real_data_trans = tf.add(real_data,0.25*true_eigen_tile)
        real_data_trans = tf.reshape(real_data_trans, [args.BATCH_SIZE, args.OUTPUT_DIM])

        
        fake_data = Generator(args.BATCH_SIZE)
        
        
        
        fake_eigen = Eigener(fake_data)
        fake_eigen_tile = tf.tile(fake_eigen[:,:,np.newaxis,np.newaxis],[1,3,64,64])#128*3*64*64        
        
        
        
        fake_data_trans = tf.add(0.25*fake_eigen_tile,fake_data)        
        fake_data_trans = tf.reshape(fake_data_trans, [-1, args.OUTPUT_DIM])
        
        
        
        alter_eigen = Eigener(real_data)
        alter_eigen_tile = tf.tile(alter_eigen[:,:,np.newaxis,np.newaxis],[1,3,64,64])#128*3*64*64                
                
        
        
        real_data_trans_alter = tf.add(real_data,0.25*alter_eigen_tile)
        real_data_trans_alter = tf.reshape(real_data_trans_alter, [-1, args.OUTPUT_DIM])
        
        #real_data = tf.reshape(real_data, [-1, args.OUTPUT_DIM])
        #fake_data = tf.reshape(fake_data, [-1, args.OUTPUT_DIM])
        
        
        
        #disc_real = Discriminator(real_data)
        #disc_fake = Discriminator(fake_data)
        
        eigen_cost = tf.reduce_mean(tf.square(true_eigen-alter_eigen))
        
        
        
        #disc_alter = Discriminator(real_data_trans_alter)
        
        
        
        disc_real = Discriminator(real_data_trans)
        disc_fake = 0.8*Discriminator(fake_data_trans) + 0.2*Discriminator(real_data_trans_alter)
        
        
        
        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')
        eigen_params = lib.params_with_name('Eigener')
        

        
        if args.MODE == 'wgan-gp':
            
            
            
            gen_cost = -tf.reduce_mean(disc_fake)
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        
        
        
            alpha = tf.random_uniform(
                shape=[args.BATCH_SIZE,1], 
                minval=0.,
                maxval=1.
            )
            
            
            
            differences = fake_data_trans - real_data_trans
            interpolates = real_data_trans + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += args.LAMBDA*gradient_penalty
        
        
        
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-3, 
                beta1=0.5,
                beta2=0.9
            ).minimize(gen_cost, var_list=gen_params)
            
            disc_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-3, 
                beta1=0.5, 
                beta2=0.9
            ).minimize(disc_cost, var_list=disc_params)
            
            eigen_train_op_gen = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5, 
                beta2=0.9
            ).minimize(gen_cost, var_list=eigen_params)                   
            
           
            
            eigen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5, 
                beta2=0.9
            ).minimize(eigen_cost, var_list=eigen_params)            
        
            clip_disc_weights = None
        

        
    
        
        
        # For saving samples
        fixed_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples= Generator(args.BATCH_SIZE, noise=fixed_noise)          
        
        def generate_image(iteration):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples, args.model_dir +'/samples_{}.png'.format(iteration))
        
        # Dataset iterator
        train_gen = load_image_eigen.load(args.BATCH_SIZE, data_dir=args.DATA_DIR)
        def inf_train_gen():
            while True:
                for (images,eigenvalues) in train_gen():
                    yield images,eigenvalues
        
        # Save a batch of ground-truth samples
        _x,_xeigen = inf_train_gen().__next__()
       # print(_xeigen)
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:args.BATCH_SIZE]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((args.BATCH_SIZE, 3, 64, 64)), 'samples_groundtruth.png')
        
        saver = tf.train.Saver(max_to_keep=80)
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        

        if args.infer=='True':

            print('stop')
            sys.exit()
            
        


 
        gen = inf_train_gen()   



        if index < args.EIGEN_ITERS:
            for iteration in range(args.EIGEN_ITERS):
                start_time = time.time()
           
    
    
    
                _data,_eigen = gen.__next__()
    
    
                        
                _eigen_cost, _ = session.run([eigen_cost,eigen_train_op],feed_dict={real_data_conv:_data,true_eigen:_eigen})            
        
                lib.plot.plot('train eigen cost', _eigen_cost)
                lib.plot.plot('time', time.time() - start_time)
                
                print("iter: %d   eigen_cost: %f"%(index,_eigen_cost))
                # Calculate dev loss and generate samples every 100 iters
                if index % 10 == 9:
    
                    saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                    _true_eigen,_alter_eigen = session.run([true_eigen,alter_eigen],feed_dict={real_data_conv:_data,true_eigen:_eigen})
                    print('true_eigen')
                    print(_true_eigen)
                    print('alter_eigen')
                    print(_alter_eigen)
                    print('eigen')
                    print(np.concatenate((_true_eigen, _alter_eigen), axis=1))
                   #saver.save(session, 'wgangp_bionics' + '.cptk')
                # Write logs every 100 iters
                if (index < 5) or (index % 10 == 9):
                    lib.plot.flush()
    
        
                lib.plot.tick()
                index = index + 1





 
        for iteration in range(args.ITERS):
            start_time = time.time()
            
            
            if iteration > 0:
                _ = session.run(gen_train_op)
                _ = session.run(eigen_train_op_gen,feed_dict={real_data_conv:_data,true_eigen:_eigen})
    

            disc_iters = args.CRITIC_ITERS
            
            
            for i in range(disc_iters):
                
                _data,_eigen = gen.__next__()
                
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data_conv:_data,true_eigen:_eigen}
                )

                    
    
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
            
            print("iter: %d   disc_cost: %f"%(index,_disc_cost))
            # Calculate dev loss and generate samples every 100 iters
            if index % 10 == 0:
                _eigen_cost,_true_eigen,_alter_eigen = session.run([eigen_cost,true_eigen,alter_eigen],feed_dict={real_data_conv:_data,true_eigen:_eigen})
                print('true_eigen')
                print(_true_eigen)
                print('alter_eigen')
                print(_alter_eigen)
                print('eigen')
                print(np.concatenate((_true_eigen, _alter_eigen), axis=1))

                print("iter: %d   eigen_cost: %f"%(index,_eigen_cost))
                
                generate_image(index)
                saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                _true_eigen,_alter_eigen = session.run([true_eigen,alter_eigen],feed_dict={real_data_conv:_data,true_eigen:_eigen})

               #saver.save(session, 'wgangp_bionics' + '.cptk')
            # Write logs every 100 iters
            if (index < 5) or (index % 10 == 0 ):
                lib.plot.flush()

    
            lib.plot.tick()
            index = index + 1

