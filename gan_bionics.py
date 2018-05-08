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
        if dim==64:
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
        if dim==128:
            output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*16*dim, noise)
                    
            output = tf.reshape(output, [-1, 16*dim, 4, 4])
        
            output = nonlinearity(output)
            
            output = lib.ops.deconv2d.Deconv2D('Generator.1', 16*dim, 8*dim, 5, output)
        
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
        output = tf.reshape(inputs, [-1, 3, args.DIM, args.DIM])
    
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        if dim==64:
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

        if dim==128:
            output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
            output = nonlinearity(output)
        
            output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
        
            output = nonlinearity(output)
        
            output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        
            output = nonlinearity(output)
        
            output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        
            output = nonlinearity(output)

            output = lib.ops.conv2d.Conv2D('Discriminator.5', 8*dim, 16*dim, 5, output, stride=2)
        
            output = nonlinearity(output)
            
            output = tf.reshape(output, [-1, 4*4*16*dim])
            output = lib.ops.linear.Linear('Discriminator.Output', 4*4*16*dim, 1, output)
            
        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()
    
        return tf.reshape(output, [-1])    
    
    real_data_conv = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, 3, args.DIM, args.DIM])
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

    train_gen = load_image.load(args.BATCH_SIZE, data_dir=args.DATA_DIR, dim=args.DIM,num=args.img_num)
    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images   
                
    gen = inf_train_gen()                
    # Train loop
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        

        # For saving samples
        fixed_noise = tf.random_uniform([args.BATCH_SIZE, 128], minval=-1.0, maxval=1.0, dtype=tf.float32, seed=1, name=None)
        fixed_noise_samples = Generator(args.BATCH_SIZE, noise=fixed_noise)          
        def generate_image(iteration):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            
            lib.save_images.save_images(samples.reshape((args.BATCH_SIZE, 3, args.DIM, args.DIM)), args.model_dir +'/samples_{}.png'.format(iteration))
        
        # Dataset iterator

        

        
        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1
        
        
        


 
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
            if index % 100 == 99:
                generate_image(index)
                saver.save(session, args.model_dir + '/wgangp_' + str(index) + '.cptk')
                #saver.save(session, 'wgangp_bionics' + '.cptk')
            # Write logs every 100 iters
            if (index < 5) or (index % 10 == 9):
                lib.plot.flush()

    
            lib.plot.tick()
            index = index + 1
