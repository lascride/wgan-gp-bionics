# -*- coding: utf-8 -*-
###############################################################################
#Calculating HOG Feature
###############################################################################

import tensorflow as tf
import numpy as np

BATCH_SIZE = 128

class HOGNet():
    def __init__(self, use_bin=True, NO=8, BS=8, nc=3):
        self.use_bin=True
        self.NO = NO
        self.BS = BS
        self.nc = nc
        self.use_bin = use_bin
        #self._comp_mask = self.def_comp_mask()    
    
    def comp_mask(self,mask):
        BS = self.BS
        mask = tf.transpose(mask,[0,2,3,1])
        bf = np.ones(( 2 * BS, 2 * BS, 1,1))
        bf_tf = tf.cast(tf.convert_to_tensor(bf),tf.float32)
        m_b = tf.nn.conv2d(mask, bf_tf, strides=[1,BS,BS,1], padding='SAME')


        masks = m_b > 1e-5
        return masks 
    
    def get_hog(self, x_o):
        use_bin = self.use_bin
        NO = self.NO
        BS = self.BS
        nc = self.nc
        
        x = (x_o + 1)/2
        x = tf.transpose(x,[0,2,3,1])
        Gx = np.array([[0, 0, 0], [-2, 0, 2], [0, 0, 0]]) / 2.0
        Gy = Gx.T
        
        f1_w = []
        for i in range(NO):
            t = np.pi / NO * i
            g = np.cos(t) * Gx + np.sin(t) * Gy
            gg = np.tile(g[ :, :,np.newaxis, np.newaxis], [1, 1, 1, 1])
            f1_w.append(gg)        
        f1_w = np.concatenate(f1_w, axis=3)
        G = np.concatenate([Gx[ :, :, np.newaxis, np.newaxis], Gy[ :, :, np.newaxis, np.newaxis]], axis=3)    
        G_tf = tf.cast(tf.convert_to_tensor(G),tf.float32)
        a = np.cos(np.pi / NO)
        l1 = 1/(1-a)
        l2 = a/(1-a)
        eps = 1e-3

        if nc == 3:
            #print(x.get_shape())
            x_gray = tf.expand_dims(tf.reduce_mean(x, axis=3),3)
        else:
            x_gray = x        
        f1 = tf.cast(tf.convert_to_tensor(f1_w),tf.float32)
        h0 = tf.abs(tf.nn.conv2d(x_gray, f1, strides=[1,1,1,1], padding='SAME'))
        g = tf.nn.conv2d(x_gray, G_tf, strides=[1,1,1,1], padding='SAME')
        
        if use_bin:

            gx = tf.expand_dims(g[:, :, :, 0],3)
            gy = tf.expand_dims(g[:, :, :, 1],3)    
            gg = tf.sqrt(gx * gx + gy * gy + eps)
            
            hk = tf.maximum(tf.cast(0,tf.float32), l1*h0-l2*gg)
            
            bf_w = np.zeros((NO, NO, 2*BS, 2*BS))
            b = 1 - np.abs((np.arange(1, 2 * BS + 1) - (2 * BS + 1.0) / 2.0) / BS)
            b = b[np.newaxis, :]
            bb = b.T.dot(b)
            for n in range(NO):
                bf_w[n,n] = bb

            bf_tf = tf.convert_to_tensor(bf_w)
            bf_tf = tf.cast(tf.transpose(bf_tf,[2,3,0,1]),tf.float32)
            h_f = tf.nn.conv2d(hk, bf_tf, strides=[1,BS,BS,1], padding='SAME')
            return h_f
        else:
            return g        
            