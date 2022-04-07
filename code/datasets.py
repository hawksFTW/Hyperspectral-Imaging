# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
#%matplotlib inline

import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import sys

from scipy.io import loadmat
import pandas as pd


def load_firmness_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas

	cols = ["fid", "firmness", "texture", "fungus", "weight"]
	df = pd.read_csv(inputPath, sep=",", header=0, names=cols)

	# determine (1) the unique zip codes and (2) the number of data
	# points with each zip code
	df = df.set_index("fid")

    # return the data frame
	return df


def load_produce_images(df, inputPath):
    inputImages = []
    for i in df.index.values:
        spectral = hdf5.loadmat(inputPath + i + ".jpeg")
        img = spectral["cube"]  #480 x 512
        img = cv2.resize(img, (256, 256)) ##change this to 128 x 128 or lower so NN does not crash

        inputImages.append(img)
        
    return (np.array(inputImages))


def process_attributes(df, train, test):
    continuous = ["firmness"]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
    trainX = np.hstack([trainContinuous])
    testX = np.hstack([testContinuous])
    
    return(trainX, testX)



spectral = hdf5.loadmat("C:\\Users\\Dhruv\\Documents\\Hyperspectraldata\\hsdata\\t410.jpeg.mat")
img = spectral["cube"][:,:,4]  #480 x 512
img1 = cv2.resize(img, (256, 256))
cv2.imshow("Org Image",img)
cv2.waitKey(0)
cv2.imshow("Resize Image",img1)
cv2.waitKey(0)



