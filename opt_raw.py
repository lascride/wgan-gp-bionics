# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:41:37 2018

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
import tflib.plot_opt
import argparse
import load_image
import PIL.Image as Image
import cv2
import scipy.misc
from scipy.misc import imsave
from skimage import feature as ft
import tf_hog 
import hog

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
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=100)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=64*64*3)
    parser.add_argument('--output_lenth', dest='output_lenth', help='lenth of the output images',type=int, default=64)
    parser.add_argument('--img_num', dest='img_num', help='the number of the output images', type=int, default=4096)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--opt_dir', dest='opt_dir', type=str, default='opt',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    parser.add_argument('--to_do', dest='to_do', help='train or opt', default='train', type=str)
    parser.add_argument('--nc', dest='nc', help='the number of channels', type=int, default=3)
    parser.add_argument('--npx', dest='npx', help='64*64', type=int, default=64)


    parser.add_argument('--edge', dest='edge', help='whether to consider egde', default='No', type=str)
    parser.add_argument('--z0', dest='z0', help='whether to consider z0', default='No', type=str)
    parser.add_argument('--input_color_name', dest='input_color_name', help='input color image name', default='input_color')
    parser.add_argument('--input_color_mask_name', dest='input_color_mask_name', help='input color mask name', default='input_color_mask')
    parser.add_argument('--input_edge_name', dest='input_edge_name', help='input edge image name', default='input_edge')

    
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
            


    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

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
    
    
    
    def get_hog_raw(gx3):
        gx_gray = np.mean(gx3,axis=1)
        print(gx_gray.shape)
        print(gx_gray[1,:,:].shape)
        #imsave(args.opt_dir+'/test_hop'+'.png',gx_gray[1,:,:])

        gx_edge = []
        
        for i in range(args.BATCH_SIZE):
            Hog = hog.Hog_descriptor(gx_gray[i,:,:], cell_size=8, bin_size=8)
            vector, image = Hog.extract()
            #print(image.shape)
            imsave(args.opt_dir+'/gray_'+str(i)+'.png',gx_gray[i,:,:])  
            imsave(args.opt_dir+'/test_'+str(i)+'.png',image)
            gx_edge.append(image[np.newaxis,np.newaxis,:])
   
        gx_edge = np.concatenate(gx_edge, axis=0)
        return gx_edge
    
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
        
        

        real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, 64, 64])
        real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [args.BATCH_SIZE, args.OUTPUT_DIM])
        
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
        fixed_noise = tf.random_uniform([128, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples = Generator(128, noise=fixed_noise)          
        def generate_image(iteration):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples.reshape((128, 3, 64, 64)), args.model_dir +'/samples_{}.png'.format(iteration))
        
        # Dataset iterator
        train_gen = load_image.load(args.BATCH_SIZE, data_dir=args.DATA_DIR)
        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images
        
        # Save a batch of ground-truth samples
        _x = inf_train_gen().__next__()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:args.BATCH_SIZE]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((args.BATCH_SIZE, 3, 64, 64)), 'samples_groundtruth.png')
        
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        if args.to_do == "opt":

            x_c_o = tf.placeholder(tf.float32, shape=[64, 64, 3])
            m_c_o = tf.placeholder(tf.float32, shape=[64, 64, 1])        
            x_c = transform(x_c_o[np.newaxis, :], 3)
            m_c = transform_mask(m_c_o[np.newaxis, :])
    
            shape = [args.BATCH_SIZE, 1, 1, 1]
            x_c = tf.tile(x_c, shape)
            m_c = tf.tile(m_c, shape)
      
            
            if args.edge == 'Yes':
                x_e_o = tf.placeholder(tf.float32, shape=[64, 64, 3])
                m_e_o = tf.placeholder(tf.float32, shape=[64, 64, 1]) 
                
                x_e = transform(x_e_o[np.newaxis, :], 3)
                m_e = transform_mask(m_e_o[np.newaxis, :])
                
  
                x_e = tf.tile(x_e, [args.BATCH_SIZE, 1, 1, 1])
                m_e = tf.tile(m_e, [args.BATCH_SIZE, 1, 1, 1])
    
            if args.z0 == 'Yes':
                z0 = tf.placeholder(tf.float32, shape=[1,128])
                
            
            z = tf.Variable(tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None), name="z")  
           
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    session.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
    
            init_new_vars_op = tf.initialize_variables(uninit_vars)
            #print("uninit")
            #print(uninit_vars)
            session.run(init_new_vars_op)           
           
           
            z_t = tf.nn.tanh(z)   

            gx = Generator(args.BATCH_SIZE, noise=z_t)
            gx3 = tf.reshape((gx+1.)*(255.99/2),[args.BATCH_SIZE, 3, 64, 64]) 
            gx3_alter = gx3
            gx3 = transform(gx3, 3,trans='no')
            mm_c = tf.tile(m_c, [1, int(gx3.shape[1]), 1, 1])#tile gray to rgb
            

            color_all = tf.reduce_mean(tf.square(gx3 - x_c) * mm_c, axis=[1, 2, 3]) / (tf.reduce_mean(m_c, axis=[1, 2, 3]) + 1e-5)
            
            
            if args.edge == 'Yes':
                tf_hog = tf_hog.HOGNet(use_bin=True, NO=8, BS=8, nc=3)
                gx_edge = tf_hog.get_hog(gx3)
                x_edge = tf_hog.get_hog(x_e)
                m_edge = tf_hog.comp_mask(m_e)
                #gx_edge = get_hog_raw(gx3.eval())
                #print(gx_edge.shape)
                #print(gx3.eval().shape)
                #assign_op = tf.assign(gx_edge, tf.convert_to_tensor(get_hog(gx3.eval()),dtype=tf.float32))


                m_edge = tf.cast(m_edge,tf.float32)
                print("m_edge")
                print(m_edge.shape)
                print("gx_edge")
                print(gx_edge.shape)    
                print("x_edge")
                print(x_edge.shape)                 
                mm_e = tf.tile(m_edge, [1, 1, 1, int(gx_edge.shape[3])])
                
                #sum_e = tf.reduce_sum(tf.abs(mm_e))
                #sum_x_edge = tf.reduce_sum(tf.abs(x_edge))
                edge_all = tf.reduce_mean(tf.square(x_edge - gx_edge) * mm_e, axis=[1, 2, 3]) / (tf.reduce_mean(m_edge, axis=[1, 2, 3]) + 1e-5)            
            else:
                edge_all = 0
            

            

                
            if args.z0 == 'Yes':
                z_const = 5.0
                init_all = tf.reduce_mean(tf.square(z0 - z)) * z_const
            else:
                init_all = 0
        

            real_all = -Discriminator(gx)

  
        
            cost_all = color_all + edge_all + 0.05*real_all +5.0 * init_all
        
            cost = tf.reduce_sum(cost_all)

        
            invert_train_op = tf.train.AdamOptimizer(
                             learning_rate=0.1, 
                             beta1=0.9
                         ).minimize(cost, var_list=[z])
    
            
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    session.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
    
            init_new_vars_op = tf.initialize_variables(uninit_vars)
            #print("uninit")
            #print(uninit_vars)
            session.run(init_new_vars_op)    

        
        
        if args.to_do == "train":
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
                        feed_dict={real_data_conv:_data}
                    )
                    if clip_disc_weights is not None:
                        _ = session.run(clip_disc_weights)
        
                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('time', time.time() - start_time)
                
                print("iter: %d   disc_cost: %f"%(index,_disc_cost))
                # Calculate dev loss and generate samples every 100 iters
                if index % 20 == 19:
                    generate_image(index)
                    
                # Write logs every 100 iters
                if (index < 5) or (index % 100 == 99):
                    lib.plot.flush()
                    saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
        
                lib.plot.tick()
                index = index + 1
                
        if args.to_do == "opt":
            #writer = tf.summary.FileWriter("logs", session.graph)
            im_color = preprocess_image('./pics/'+args.input_color_name+'.png', args.npx)
         
            #im_color_mask = preprocess_image(args.input_color_mask, args.npx)
            imsave(args.opt_dir+'/im_color'+'.png',im_color)
            #cv2.imwrite(args.opt_dir+'/im_colormask'+'.png',im_color_mask)
            im_color_mask_mask = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
            #cv2.imwrite(args.opt_dir+'/im_colormask_mask1'+'.png',im_color_mask_mask)
            ret,im_color_mask_mask = cv2.threshold(im_color_mask_mask,1,255,cv2.THRESH_BINARY)
            #cv2.imwrite(args.opt_dir+'/im_colormask'+'.png',im_color_mask_mask)
            im_color_mask_mask = cv2.cvtColor(im_color_mask_mask, cv2.COLOR_GRAY2RGB)
            imsave(args.opt_dir+'/im_colormask'+'.png',im_color_mask_mask)
            #im_color = np.rot90(im_color,2)
            #cv2.imwrite(args.opt_dir+'/im_color_rot'+'.png',im_color)
            
            #im_color_mask_mask = np.rot90(im_color_mask_mask,2)
            #cv2.imwrite(args.opt_dir+'/im_colormask_rot'+'.png',im_color_mask_mask)

    
            #print(im_color_mask_mask.shape)

           # color = Image.open(args.input_color)
            #color_mask = Image.open(args.input_color_mask)
           # color.save(args.opt_dir+'/color'+'.png')
          #  color_mask.save(args.opt_dir+'/colormask'+'.png')
           # print("hahaahahha")
            if args.edge == 'Yes':
                im_edge = preprocess_image('./pics/'+args.input_edge_name+'.png', args.npx)
                im_edge_mask = im_edge[...,[0]]
                #print(im_edge_mask.shape)
                #im_edge_mask = im_edge[...,[0]].reshape((64,64))
                #print(im_edge_mask.shape)

                #Hog_input = hog.Hog_descriptor(im_edge_mask, cell_size=8, bin_size=8)           
                #_, im_edge_edge = Hog_input.extract()
                imsave(args.opt_dir+'/im_edge'+'.png',im_edge)
                imsave(args.opt_dir+'/im_edge_mask'+'.png',im_edge_mask.reshape((64,64)))
                #imsave(args.opt_dir+'/im_edge_edge'+'.png',im_edge_edge)
                #im_edge_edge = im_edge_edge[:,:,np.newaxis]
                #im_edge_mask = im_edge_mask[:,:,np.newaxis]
                
               # print(im_edge_edge.shape)
                #print(im_edge_mask.shape)
            for iteration in range(args.ITERS):
                start_time = time.time()


                if args.edge == 'Yes':
                    feed_dict = {x_c_o : im_color, m_c_o : im_color_mask_mask[... ,[0]], x_e_o:im_edge, m_e_o: im_edge_mask}
                else:
                    feed_dict = {x_c_o : im_color, m_c_o : im_color_mask_mask[... ,[0]]}

                #test = session.run(x_e,feed_dict={x_e_o:im_edge_edge,m_e_o: im_edge_mask})
                #print(test.shape)
                #_gx_edge = session.run(gx_edge)
                #print(_gx_edge)
                if args.edge == 'Yes':
                    _x_c,_gx3, _m_c_o, _color_all,_real_all, _m_edge,_edge_all,_z_t,_gx, _cost, _cost_all, _ = session.run([x_c,gx3, m_c_o,color_all,real_all,m_edge,edge_all,z_t,gx,cost,cost_all, invert_train_op], feed_dict=feed_dict)
                    print('colorall')                    
                    print(_color_all)                    
                    print('edgeall')                    
                    print(_edge_all)
                    print('costall')
                    print(_cost_all)
                    print('realall')
                    print(_real_all)
                    #print('m_c_o')
                    #print(_m_c_o.reshape(64,64)) 
                    #print('gx3')
                    #print(_gx3[0,0,:,:].reshape(64,64)) 
                    #print('x_c')
                    #print(_x_c[0,0,:,:].reshape(64,64)) 
                else:
                    _z_t,_gx, _cost, _cost_all, _ = session.run([z_t,gx,cost,cost_all, invert_train_op], feed_dict=feed_dict)

                #session.run(assign_op)
                order_all = np.argsort(_cost_all)
                order_color = np.argsort(_color_all)
                order_edge = np.argsort(_edge_all)
                order_real = np.argsort(_real_all)                

    
                lib.plot_opt.plot('cost', _cost)
                lib.plot_opt.plot('time', time.time() - start_time)    
               

                print("iter: %d ; cost_all: %f"%(iteration,_cost))
                # Calculate dev loss and generate samples every 100 iters
                if (iteration % 10 == 9) or (iteration==0):
                    lib.plot_opt.flush()
                   # print(_m_edge[0,:,:,0])
                    imsave(args.opt_dir+'/sketch_edge_mask'+'.png',_m_edge[0,:,:,0].reshape((_m_edge.shape[1],_m_edge.shape[2])))
                   # print(_cost_all[order_all])
                    #print(_edge_all)
                    #print(_gx_edge)
                    #_ggx = session.run(Generator(128))
                    #_ggx = ((_ggx+1.)*(255.99/2)).astype('int32')
                    _gx = ((_gx+1.)*(255.99/2)).astype('int32')
                    _gx_raw = tf.reshape(_gx,[args.BATCH_SIZE, 3, 64, 64]).eval()
                    _gx_all = tf.reshape(_gx[order_all],[args.BATCH_SIZE, 3, 64, 64]).eval()
                    _gx_color = tf.reshape(_gx[order_color],[args.BATCH_SIZE, 3, 64, 64]).eval()
                    _gx_edge = tf.reshape(_gx[order_edge],[args.BATCH_SIZE, 3, 64, 64]).eval()
                    _gx_real = tf.reshape(_gx[order_real],[args.BATCH_SIZE, 3, 64, 64]).eval()                    
                    #lib.save_images.save_images(_gx3.astype('int32'), args.opt_dir+'/aaaa_'+args.input_color_name+'_'+str(iteration)+'.png')
                    lib.save_images.save_images(_gx_raw, args.opt_dir+'/aaraw_'+args.input_color_name+'_'+str(iteration)+'.png')
                    lib.save_images.save_images(_gx_all, args.opt_dir+'/all_'+args.input_color_name+'_'+str(iteration)+'.png')
                    lib.save_images.save_images(_gx_color, args.opt_dir+'/color_'+args.input_color_name+'_'+str(iteration)+'.png')
                    lib.save_images.save_images(_gx_edge, args.opt_dir+'/edge_'+args.input_color_name+'_'+str(iteration)+'.png')
                    lib.save_images.save_images(_gx_real, args.opt_dir+'/real_'+args.input_color_name+'_'+str(iteration)+'.png')

                    #lib.save_images.save_images(_ggx.reshape((args.BATCH_SIZE, 3, 64, 64)), 'opt+str_1.png') 
    
                lib.plot_opt.tick()
                    
    
    
     
                
        
