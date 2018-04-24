import os, sys
sys.path.append(os.getcwd())

import time
import functools

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
import tflib.ops.layernorm
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
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='models_good',
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
        
    def Normalize(name, axes, inputs):
        if ('Discriminator' in name) and (args.MODE == 'wgan-gp'):
            if axes != [0,2,3]:
                raise Exception('Layernorm over non-standard axes is unsupported')
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        else:
            return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

    def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        return output
    
    def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output
    
    def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0,3,1,2])
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
        """
        resample: None, 'down', or 'up'
        """
        if resample=='down':
            conv_shortcut = MeanPoolConv
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        elif resample=='up':
            conv_shortcut = UpsampleConv
            conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        elif resample==None:
            conv_shortcut = lib.ops.conv2d.Conv2D
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')
    
        if output_dim==input_dim and resample==None:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)
    
        output = inputs
        output = Normalize(name+'.BN1', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
        output = Normalize(name+'.BN2', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
    
        return shortcut + output


    
    def Generator(n_samples, noise=None, dim=args.DIM, nonlinearity=tf.nn.relu):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])
    
        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])
    
        output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')
    
        output = Normalize('Generator.OutputN', [0,2,3], output)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
        output = tf.tanh(output)
    
        return tf.reshape(output, [-1, args.OUTPUT_DIM])
    
    
    def Discriminator(inputs, dim=args.DIM):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)
    
        output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')
    
        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)
    
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
        fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
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
        
        
        #train_gen, dev_gen, test_gen = lib.mnist.load(args.BATCH_SIZE, args.BATCH_SIZE)

        
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        index = 0
        if args.restore_index:
            saver.restore(session,args.model_dir+"/wgangp_"+str(args.restore_index)+".cptk")
            index = index + args.restore_index + 1
        
        
        
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