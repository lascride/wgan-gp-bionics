# -*- coding: utf-8 -*-
"""
Created on Fri May  4 00:26:24 2018

@author: lenovo
"""
import theano
import theano.tensor as T
from time import time
from lib import updates, HOGNet
from lib.rng import np_rng
from lib.theano_utils import floatX, sharedX
import numpy as np

def preprocess_image(img_path, npx):
    im = cv2.imread(img_path, 1)
    if im.shape[0] != npx or im.shape[1] != npx:
        out = cv2.resize(im, (npx, npx))
    else:
        out = np.copy(im)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
    
im_edge = preprocess_image('e:/git/wgan-gp-bionics/pics/'+'raw_edge'+'.png', 3)

hog = HOGNet.HOGNet(use_bin=True, NO=8, BS=BS, nc=self.nc)
gx_edge = hog.get_hog(gx3)