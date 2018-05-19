"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import cv2


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    

    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw

        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)
    
    
def save_samples(X, _im_color, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')
    color_label =cv2.resize(_im_color, (128, 128))
    X = X[0:16]
    
    rows = 2


    nh, nw = rows, 8


    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = 64,64
        img = np.zeros((h*nh, w*(nw+2)+16, 3))

    img[0:128, 0:128] = color_label
    img[0:128, 128:144] = 255.99*np.ones((128, 16, 3)).astype('uint8')
    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw

        img[j*h:j*h+h, ((i+2)*w+16):((i+2)*w+w+16)] = x
        

    imsave(save_path, img)    
    
def save_images_discrete(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    

    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw

        img[j*h:j*h+h, i*w:i*w+w] = x
        imsave(save_path+'/sample_'+str(n)+'.png', img)

    