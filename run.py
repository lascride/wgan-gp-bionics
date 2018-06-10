# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:56:08 2018

@author: lenovo
"""

import os, sys
import numpy as np
import pandas as pd
import itertools

a = ['raw_14','raw_15','raw_16']
b = [0.0,0.001,0.002,0.005,0.01]
c = itertools.product(a,b)

c = np.array(list(c))
print(c)


for input_color_name,weight_real in c:
        print(input_color_name,weight_real,end=' ')
        
for input_color_name,weight_real in c:
    os.system('python -u '+'opt.py'+' --BATCH_SIZE '+'128'+' --input_edge_name '+'blank'+' --input_color_name '+input_color_name+' --weight_real '+weight_real+' --restore_index '+'249999'+' --ITERS '+'100')