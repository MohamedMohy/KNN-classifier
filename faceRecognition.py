#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:51:13 2018

@author: Zyad Shokry & Kareem Abd-El-Salam
"""

import matplotlib.image as img
import numpy as np
from numpy import linalg as LA

folder="/home/mohamed/machine-learning/KNN/orl_faces/"

imgMat=np.zeros((0,10304))
temp=np.arange(1,41,1)
label_matrix=np.array([[temp[i]]*10 for i in range (temp.size)])
label_matrix=label_matrix.flatten()

for j in range(1,41):
    direction=folder+"s"+str(j)+"/"
    for i in range (1,11):
        directory=direction+str(i)+ ".pgm"
        image = (img.imread(directory))
        imageVect=np.asmatrix(image.flatten())
        imgMat=np.concatenate((imgMat,imageVect))

training_data_matrix=imgMat[0:400:2]
test_data_matrix=imgMat[1:400:2]

    
