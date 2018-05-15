# -*- coding: utf-8 -*-
"""
Created on Sun May  6 21:31:36 2018

@author: lenovo
"""

import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
np.set_printoptions(threshold=np.inf)  
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist

import tflib.plot_opt

import argparse

import cv2
from scipy.misc import imsave
import tf_hog 



def parse_args():
    parser = argparse.ArgumentParser(description='cut images')
    parser.add_argument('--output_path', dest='output_path', help='the output path', default='e:/project/project/image/input_3_64_10000_rot/10', type=str)
    parser.add_argument('--DATA_DIR', dest='DATA_DIR', help='the input path', default='e:/project/project/image/input_3_64_10000_rot/10', type=str)
    parser.add_argument('--color_mode', dest='color_mode', help='rgb or gray-scale', default='rgb', type=str)
    parser.add_argument('--DIM', dest='DIM', help='Model dimensionality',type=int, default=64)
    parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', help='Batch size',type=int, default=128)
    parser.add_argument('--CRITIC_ITERS', dest='CRITIC_ITERS', help='For WGAN and WGAN-GP, number of critic iters per gen iter',type=int, default=5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='Gradient penalty lambda hyperparameter',type=int, default=10)
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=100)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=64*64*3)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--opt_dir', dest='opt_dir', type=str, default='opt',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    parser.add_argument('--nc', dest='nc', help='the number of channels', type=int, default=3)
    parser.add_argument('--npx', dest='npx', help='64*64', type=int, default=64)


    parser.add_argument('--z0', dest='z0', help='whether to consider z0', default='No', type=str)
    parser.add_argument('--input_color_name', dest='input_color_name', help='input color image name', default='blank')
    parser.add_argument('--input_edge_name', dest='input_edge_name', help='input edge image name', default='blank')

    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.opt_dir):
        os.makedirs(args.opt_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

        
    def transform(x, nc=3, trans = 'yes'):
        if trans == 'yes':
            if nc == 3:
                return 2*((tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.)-.5)
            else:
                return (tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.)
        else:
            if nc == 3:
                return 2*((tf.cast(x, tf.float32)/255.)-.5)
            else:
                return (tf.cast(x, tf.float32)/255.)            
            
            

    def transform_mask( x, trans = 'yes'):
        if trans == 'yes':
            return (tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.) 
        else:
            return (tf.cast(x, tf.float32)/255.) 
            




    def preprocess_image(img_path, npx):
        im = cv2.imread(img_path, 1)
        if im.shape[0] != npx or im.shape[1] != npx:
            out = cv2.resize(im, (npx, npx))
        else:
            out = np.copy(im)
    
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    
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
        


        # For reloading samples
        fixed_noise = tf.random_uniform([128, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples = Generator(128, noise=fixed_noise)  
        fixed_noise_samples_disc = Discriminator(fixed_noise_samples)        

        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1        


        #processing color constriants
        x_c_o = tf.placeholder(tf.float32, shape=[64, 64, 3])
        m_c_o = tf.placeholder(tf.float32, shape=[64, 64, 1])        
        x_c = transform(x_c_o[np.newaxis, :], 3)
        m_c = transform_mask(m_c_o[np.newaxis, :])
        x_c = tf.tile(x_c, [args.BATCH_SIZE, 1, 1, 1])
        m_c = tf.tile(m_c, [args.BATCH_SIZE, 1, 1, 1])
  
        
        #processing edge constriants
        x_e_o = tf.placeholder(tf.float32, shape=[64, 64, 3])
        m_e_o = tf.placeholder(tf.float32, shape=[64, 64, 1]) 
        x_e = transform(x_e_o[np.newaxis, :], 3)
        m_e = transform_mask(m_e_o[np.newaxis, :])
        x_e = tf.tile(x_e, [args.BATCH_SIZE, 1, 1, 1])
        m_e = tf.tile(m_e, [args.BATCH_SIZE, 1, 1, 1])


            
        #initializing z
        z = tf.Variable(tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None), name="z")  
       

        z_t = tf.nn.tanh(z)   

        gx = Generator(args.BATCH_SIZE, noise=z_t)
        gx3 = tf.reshape((gx+1.)*(255.99/2),[args.BATCH_SIZE, 3, 64, 64]) 
        gx3 = transform(gx3, 3,trans='no')
        
        #color cost
        mm_c = tf.tile(m_c, [1, int(gx3.shape[1]), 1, 1])#tile gray to rgb
        color_all = tf.reduce_mean(tf.square(gx3 - x_c) * mm_c, axis=[1, 2, 3]) / (tf.reduce_mean(m_c, axis=[1, 2, 3]) + 1e-5)
        
        
        #edge cost
        tf_hog = tf_hog.HOGNet(use_bin=True, NO=16, BS=3, nc=3)
        gx_edge = tf_hog.get_hog(gx3)
        x_edge = tf_hog.get_hog(x_e)
        m_edge = tf_hog.comp_mask(m_e)

        m_edge = tf.cast(m_edge,tf.float32)           
        mm_e = tf.tile(m_edge, [1, 1, 1, int(gx_edge.shape[3])])
        edge_all = tf.reduce_mean(tf.square(x_edge - gx_edge) * mm_e, axis=[1, 2, 3]) / (tf.reduce_mean(m_edge, axis=[1, 2, 3]) + 1e-5)            



        #real cost
        real_all = -Discriminator(gx)

  
    
        cost_all = color_all + 0.8 * edge_all + 0.01*real_all
        cost = tf.reduce_sum(cost_all)

    
        invert_train_op = tf.train.AdamOptimizer(
                         learning_rate=0.1, 
                         beta1=0.9
                     ).minimize(cost, var_list=[z])

        
        #initializing
        uninit_vars = []
        for var in tf.all_variables():
            try:
                session.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninit_vars)
        session.run(init_new_vars_op)    

    
    

            
        #processing color
        im_color = preprocess_image('./pics/'+args.input_color_name+'.png', args.npx)  
        imsave(args.opt_dir+'/im_color'+'.png',im_color)
        im_color_mask_mask = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
        ret,im_color_mask_mask = cv2.threshold(im_color_mask_mask,1,255,cv2.THRESH_BINARY)
        im_color_mask_mask = cv2.cvtColor(im_color_mask_mask, cv2.COLOR_GRAY2RGB)
        imsave(args.opt_dir+'/im_colormask'+'.png',im_color_mask_mask)

        #processing edge
        im_edge = preprocess_image('./pics/'+args.input_edge_name+'.png', args.npx)
        im_edge_mask = im_edge[...,[0]]
        imsave(args.opt_dir+'/im_edge'+'.png',im_edge)
        imsave(args.opt_dir+'/im_edge_mask'+'.png',im_edge_mask.reshape((64,64)))

        for iteration in range(args.ITERS):
            start_time = time.time()


            feed_dict = {x_c_o : im_color, m_c_o : im_color_mask_mask[... ,[0]], x_e_o:im_edge, m_e_o: im_edge_mask}



            _x_c,_gx3, _m_c_o, _color_all,_real_all, _m_edge,_edge_all,_z_t,_gx, _cost, _cost_all, _ = session.run([x_c,gx3, m_c_o,color_all,real_all,m_edge,edge_all,z_t,gx,cost,cost_all, invert_train_op], feed_dict=feed_dict)
            print('colorall')                    
            print(_color_all)                    
            print('edgeall')                    
            print(_edge_all)
            print('costall')
            print(_cost_all)
            print('realall')
            print(_real_all)            
            #get orders
            order_all = np.argsort(_cost_all)
            order_color = np.argsort(_color_all)
            order_edge = np.argsort(_edge_all)
            order_real = np.argsort(_real_all)                


            lib.plot_opt.plot('cost', _cost)
            lib.plot_opt.plot('time', time.time() - start_time)    
           

            #print("iter: %d ; cost_all: %f"%(iteration,_cost))
            
            if (iteration % 10 == 9) or (iteration==0):
                lib.plot_opt.flush()

                imsave(args.opt_dir+'/sketch_edge_mask'+'.png',_m_edge[0,:,:,0].reshape((_m_edge.shape[1],_m_edge.shape[2])))
        
                #saving images
                _gx = ((_gx+1.)*(255.99/2)).astype('int32')
                
                _gx_raw = tf.reshape(_gx,[args.BATCH_SIZE, 3, 64, 64]).eval()
                _gx_all = tf.reshape(_gx[order_all],[args.BATCH_SIZE, 3, 64, 64]).eval()
                _gx_color = tf.reshape(_gx[order_color],[args.BATCH_SIZE, 3, 64, 64]).eval()
                _gx_edge = tf.reshape(_gx[order_edge],[args.BATCH_SIZE, 3, 64, 64]).eval()
                _gx_real = tf.reshape(_gx[order_real],[args.BATCH_SIZE, 3, 64, 64]).eval()                    
                
                lib.save_images.save_images(_gx_raw, args.opt_dir+'/aaraw_'+args.input_color_name+'_'+str(iteration)+'.png')
                lib.save_images.save_images(_gx_all, args.opt_dir+'/all_'+args.input_color_name+'_'+str(iteration)+'.png')
                lib.save_images.save_images(_gx_color, args.opt_dir+'/color_'+args.input_color_name+'_'+str(iteration)+'.png')
                lib.save_images.save_images(_gx_edge, args.opt_dir+'/edge_'+args.input_color_name+'_'+str(iteration)+'.png')
                lib.save_images.save_images(_gx_real, args.opt_dir+'/real_'+args.input_color_name+'_'+str(iteration)+'.png')


            lib.plot_opt.tick()
                    
    
    
     
                
        
