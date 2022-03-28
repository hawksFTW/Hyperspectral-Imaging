# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:46:31 2021

@author: Dhruv
"""
import os 
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics 
import numpy as np


features = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/rgb_features.csv', index_col=0)
df_firmness = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/firmness.csv', index_col=0)

df = df_firmness.merge(features, on='fid', how='inner')
df_left = df_firmness.merge(features, on='fid', how='left')
df_na = df_left.loc[df_left['max0'].isnull()]


# =============================================================================
# x_data = features
# x_data = features.set_index('fid')
# y_data = df_firmness[['fid', 'firmness']]
# y_data = y_data.set_index('fid')
# 
# =============================================================================

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)

lr = LinearRegression()
lr.fit(train_x, train_y)

pred_y = lr.predict(test_x)
pred_y[:5]
test_y[:5]

rmsetest = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
print(rmsetest)

pred_x = lr.predict(train_x)

rmsetrain = np.sqrt(metrics.mean_squared_error(train_y, pred_x))
train_y["pred_x"] = pred_x
print(rmsetrain)

test_y['pred_y'] = pred_y


from sklearn.linear_model import RidgeCV
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_x, train_y)
clf.score(train_x, train_y)
clf_pred_y = clf.predict(test_x)

clfscore = np.sqrt(metrics.mean_squared_error(test_y, clf_pred_y))
print(clfscore)
test_y["pred_y"] = clf_pred_y

pred_tr = clf.predict(train_x)
clfscore_train = np.sqrt(metrics.mean_squared_error(train_y, pred_tr))
train_y['pred_tr'] = pred_tr
print(clfscore_train)

















