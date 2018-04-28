
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
#from tflib import HOGNet
import argparse
import load_image
import PIL.Image as Image
import cv2




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
    parser.add_argument('--ITERS', dest='ITERS', help='How many generator iterations to train for',type=int, default=200)
    parser.add_argument('--OUTPUT_DIM', dest='OUTPUT_DIM', help='Number of pixels in MNIST (28*28)',type=int, default=64*64*3)
    parser.add_argument('--output_lenth', dest='output_lenth', help='lenth of the output images',type=int, default=64)
    parser.add_argument('--img_num', dest='img_num', help='the number of the output images', type=int, default=4096)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models',
                        help='directory to save models') 
    parser.add_argument('--restore_index', dest='restore_index', help='the index of file that stores the model', type=int, default=None)
    parser.add_argument('--nc', dest='nc', help='the number of channels', type=int, default=3)
    parser.add_argument('--npx', dest='npx', help='64*64', type=int, default=64)


    parser.add_argument('--edge', dest='edge', help='whether to consider egde', default='No', type=str)
    parser.add_argument('--z0', dest='z0', help='whether to consider z0', default='No', type=str)
    parser.add_argument('--input_color', dest='input_color', help='input color image', default='./pics/input_color.png')
    parser.add_argument('--input_color_mask', dest='input_color_mask', help='input color mask', default='./pics/input_color_mask.png')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
        
    def preprocess_constraints( constraints):
        [im_c_o, mask_c_o, im_e_o, mask_e_o] = constraints
        im_c = transform(im_c_o[np.newaxis, :], args.nc)
        mask_c = transform_mask(mask_c_o[np.newaxis, :])
        im_e = transform(im_e_o[np.newaxis, :], args.nc)
        mask_t = transform_mask(mask_e_o[np.newaxis, :])
        mask_e = hog.comp_mask(mask_t)
        shape = [args.BATCH_SIZE, 1, 1, 1]
        im_c_t = np.tile(im_c, shape)
        mask_c_t = np.tile(mask_c, shape)
        im_e_t = np.tile(im_e, shape)
        mask_e_t = np.tile(mask_e, shape)
        return [im_c_t, mask_c_t, im_e_t, mask_e_t]
        
    def transform(x, nc=3):
        if nc == 3:
            return 2*((tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.)-.5)
        else:
            return (tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.)


    def transform_mask( x):
        return (tf.cast(tf.transpose(x,[0,3,1,2]), tf.float32)/255.) 
    
    if args.edge == 'Yes':
        BS = 4 if args.nc == 1 else 8
        hog = HOGNet.HOGNet(use_bin=True, NO=8, BS=BS, nc=args.nc)

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
        
        
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:


        x_c_o = tf.placeholder(tf.float32, shape=[64, 64, 3])
        m_c_o = tf.placeholder(tf.float32, shape=[64, 64, 1])        
        x_c = transform(x_c_o[np.newaxis, :], args.nc)
        m_c = transform_mask(m_c_o[np.newaxis, :])

        shape = [args.BATCH_SIZE, 1, 1, 1]
        x_c = tf.tile(x_c, shape)
        m_c = tf.tile(m_c, shape)
  
        
        if args.edge == 'Yes':
            x_e = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, 64, 64])
            m_e = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 1, 64, 64])

        if args.z0 == 'Yes':
            z0 = tf.placeholder(tf.float32, shape=[1,128])
            
        z = tf.Variable(tf.random_uniform([128, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=None, name=None), name="z")  

        gx = Generator(128, noise=z)
        gx3 = tf.reshape((gx+1.)*(255.99/2),[args.BATCH_SIZE, 3, 64, 64])


        mm_c = tf.tile(m_c, [1, int(gx3.shape[1]), 1, 1])#tile gray to rgb


        test = tf.reduce_mean(tf.sqrt(gx3 - x_c) * mm_c, axis=[1, 2, 3])
        color_all = tf.reduce_mean(tf.sqrt(gx3 - x_c) * mm_c, axis=[1, 2, 3]) / (tf.reduce_mean(m_c, axis=[1, 2, 3]) + 1e-5)
        cost_all = tf.reduce_sum(color_all)
        
        if args.edge == 'Yes':
            gx_edge = hog.get_hog(gx3)
            x_edge = hog.get_hog(x_e)
            mm_e = np.tile(m_e, (1, gx_edge.shape[1], 1, 1))
            edge_all = tf.reduce_mean(tf.sqrt(x_edge - gx_edge) * mm_e, axis=(1, 2, 3)) / (tf.reduce_mean(m_e, axis=(1, 2, 3)) + 1e-5)
            rec_all = color_all + edge_all * 0.2
            
        if args.z0 == 'Yes':
            z_const = 5.0
            init_all = tf.reduce_mean(tf.sqrt(z0 - z)) * z_const
            cost_all = rec_all + init_all
    
        invert_train_op = tf.train.AdamOptimizer(
                         learning_rate=2e-4, 
                         beta1=0.5
                     ).minimize(cost_all, var_list=[z])
    
    

        saver = tf.train.import_meta_graph(args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk.meta")


        session.run(tf.initialize_all_variables())

        saver.restore(session, args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")


        for iteration in range(args.ITERS):
            
            im_color = preprocess_image(args.input_color, args.npx)
            im_color_mask = preprocess_image(args.input_color_mask, args.npx)
            if args.edge == 'Yes':
                im_edge = preprocess_image(args.input_edge, args.npx)
                        
            feed_dict = {x_c_o : im_color, m_c_o : im_color_mask[... ,[0]]}

            _gx ,_cost_all, _ = session.run([gx,cost_all, invert_train_op], feed_dict=feed_dict)

 




            
            print("iter: %d   disc_cost: %f"%(iteration,_cost_all))
            # Calculate dev loss and generate samples every 100 iters
            if iteration % 20 == 19:
                _ggx = session.run(Generator(128))
                _ggx = ((_ggx+1.)*(255.99/2)).astype('int32')
                _gx = ((_gx+1.)*(255.99/2)).astype('int32')
                lib.save_images.save_images(_gx.reshape((args.BATCH_SIZE, 3, 64, 64)), 'opt.png') 
                lib.save_images.save_images(_ggx.reshape((args.BATCH_SIZE, 3, 64, 64)), 'opt_1.png') 

                


 