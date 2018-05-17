# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:39:51 2018

@author: lenovo
"""
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
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=512)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=200000)
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

    
    
 
    

    
    # Train loop
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        


        real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, 64, 64])
        real_data = 2*((tf.cast(real_data_conv, tf.float32)/255.)-.5)
        
        true_eigen = tf.placeholder(tf.float32, shape=[args.BATCH_SIZE,1])



        alter_eigen = Eigener(real_data)


        
        eigen_cost = tf.reduce_mean(tf.square(true_eigen-alter_eigen))

        eigen_params = lib.params_with_name('Eigener')
        

        eigen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(eigen_cost, var_list=eigen_params)            
    
        clip_disc_weights = None
    

        
    
        

        # Dataset iterator
        train_gen = load_image_eigen.load(args.BATCH_SIZE, data_dir=args.DATA_DIR)
        def inf_train_gen():
            while True:
                for (images,eigenvalues) in train_gen():
                    yield images,eigenvalues
        


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

 
        for iteration in range(args.ITERS):
            start_time = time.time()
       

            disc_iters = args.CRITIC_ITERS
            

            _data,_eigen = gen.__next__()


                    
            _eigen_cost, _ = session.run([eigen_cost,eigen_train_op],feed_dict={real_data_conv:_data,true_eigen:_eigen})            
    
            lib.plot.plot('train eigen cost', _eigen_cost)
            lib.plot.plot('time', time.time() - start_time)
            
            print("iter: %d   disc_cost: %f"%(index,_eigen_cost))
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


