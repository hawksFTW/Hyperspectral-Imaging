# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 08:54:46 2022

@author: Dhruv
"""

import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import sys

from scipy.io import loadmat
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

hs_path = './Documents/Hyperspectral_Imaging_Firmness/data/hsdata16/' #specify the path of the directory that has all the hyperspectral mat files
#file_list_1 = []
fileCount = 0
df = pd.DataFrame()
for filename in os.listdir(hs_path):
    fileCount += 1
    spectral = pd.read_csv(hs_path  + filename)
    spectral = spectral[spectral.columns[~spectral.columns.isin(['Name', 'Color', 'x', 'y'])]]
    spectral = spectral.rename(columns={x:y for x,y in zip(spectral.columns,range(0,len(spectral.columns)))})
    spectral["fid"] = 't' + (filename.partition('-')[2]).partition('-')[0]
    if fileCount == 1:
        df = spectral
    else:
        df = df.append(spectral, ignore_index=True)
   
df.to_csv(hs_path + 'hs16_features.csv', index=False)
      
########check how specific channel for image looks like
# =============================================================================
# filename = "reconCube410.txt"      
# spectral = np.loadtxt(hs_path  + filename)
# spectral = spectral.reshape((50,88,88))
# img = spectral[5,:,:]
# #cv2.imshow("Spectral Channel 5",img)
# #cv2.waitKey(0)      
# im1 = img*0.2
# cv2.imshow("Spectral Channel 5(0.6)",im1)
# cv2.waitKey(0)   
# cv2.imwrite("t410_channel5.png",img);
# =============================================================================

    
    






