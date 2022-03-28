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
#import xgboost 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV

hs16_features = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/hs16_features.csv', index_col=[0])
hs50_features = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/hs50_features.csv', index_col=[0])
hs31_features = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/hs31_features.csv', index_col=[0])    ###hs_31 features, file needs to be regenerated
rgb_features = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/rgb_features.csv', index_col=[0])
df_firmness = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/firmness_all.csv', index_col=[0])
#valid_hs31 = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/valid_hs31.csv', index_col=[0])  
valid12_hs31 = pd.read_csv('./Documents/Hyperspectral_Imaging_Firmness/data/valid12_hs31.csv', index_col=[0])


############Spectral reconstruction using existing research
df = df_firmness.merge(hs31_features, on='fid', how='inner')
#df_left = df_firmness.merge(hs31_features, on='fid', how='left')
#df_na = df_left.loc[df_left['max0'].isnull()]


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

#plot_learning_curve(lr, "Linear Regression Loss Curve", , y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#rmsetest = np.sqrt(metrics.mean_squared_error(test_y, pred_y))
#print(rmsetest)
pred_x = lr.predict(train_x)
#rmsetrain = np.sqrt(metrics.mean_squared_error(train_y, pred_x))
train_y["pred_x"] = pred_x
#print(rmsetrain)

test_y['pred_y'] = pred_y


###########RidgeCV regression ################################
# =============================================================================
# X= train_x
# y = train_y
# # define model
# model = Ridge()
# # define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # define grid
# grid = dict()
# grid['alpha'] = np.arange(0, 1, 0.01)
# # define search
# search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # perform the search
# results = search.fit(X, y)
# # summarize
# print('MAE: %.3f' % results.best_score_)
# print('Config: %s' % results.best_params_)
# =============================================================================

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_x, train_y)
#clf = Ridge(alpha=0.99)
clf.fit(train_x, train_y)
clf.score(train_x, train_y)
clf_pred_y = clf.predict(test_x)
valid_pred_y = clf.predict(valid12_hs31[valid12_hs31.columns[~valid12_hs31.columns.isin(['fid'])]])

#clfscore = np.sqrt(metrics.mean_squared_error(test_y, clf_pred_y))
#print(clfscore)
test_y["pred_y"] = clf_pred_y

##validation test 
#v1 = pd.read_csv("./Documents/Hyperspectral_Imaging_Firmness/data/" + 'v1_hs31.csv', index_col=[0])
valid12_hs31 = valid12_hs31[valid12_hs31.columns[~valid12_hs31.columns.isin(['fid'])]]
valid_preds = pd.DataFrame(np.exp(clf.predict(valid12_hs31)))


#pred_tr = clf.predict(train_x)
#clfscore_train = np.sqrt(metrics.mean_squared_error(train_y, pred_tr))
#train_y['pred_tr'] = pred_tr
#print(clfscore_train)

########## RGB ##############
df = df_firmness.merge(rgb_features, on='fid', how='inner')
df_left = df_firmness.merge(rgb_features, on='fid', how='left')
df_na = df_left.loc[df_left['max0'].isnull()]

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)

lr = LinearRegression()
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)

test_y['pred_y'] = pred_y

#clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_x, train_y)
#rcv = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
clf.score(train_x, train_y)
clf_pred_y = clf.predict(test_x)

test_y["pred_y"] = clf_pred_y


########## Hyperspectral 50 channels ##############
df_firmness = pd.read_csv("./Documents/Hyperspectral_Imaging_Firmness/data/firmness_hs50_omit.csv", index_col = [0])
df = df_firmness.merge(hs50_features, on='fid', how='inner')
omit_files = pd.read_csv("./Documents/Hyperspectral_Imaging_Firmness/data/omit_files.csv", index_col=[0])
omit_files.reset_index(inplace = True)
#df_left = df_firmness.merge(hs50_features, on='fid', how='left')
#df_na = df_left.loc[df_left['max0'].isnull()]
df.reset_index(inplace = True)
df = (df[~df.fid.isin(omit_files.filename)])
df[['fid', 'firmness']].to_csv('./Documents/Hyperspectral_Imaging_Firmness/data/firmness_hs50_omit.csv')

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)

lr = LinearRegression()
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)
test_y['pred_y'] = pred_y
tr_pred_y = lr.predict(train_x)
train_y["pred_y"] = tr_pred_y

from sklearn.linear_model import RidgeCV
train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_x, train_y)
clf.score(train_x, train_y)
clf_pred_y = clf.predict(test_x)
test_y["pred_y"] = clf_pred_y


########## Hyperspectral 15/16 channels ##############
#df_firmness_50 = pd.read_csv("C:/Users/Nalina/Documents/Hyperspectraldata/data/firmness_hs50.csv")
#hs16_features = hs16_features[hs16_features.columns[~hs16_features.columns.isin(['15'])]]
hs16_features.reset_index(inplace=True)
hs16_features.iloc[:,0:16] = hs16_features.iloc[:,0:16].div(hs16_features['0'], axis=0)
df = df_firmness.merge(hs16_features, on='fid', how='inner')
#df[['fid', 'firmness']].to_csv("C:/Users/Nalina/Documents/Hyperspectraldata/data/firmness_hs15.csv", index=False)
#df_left = df_firmness.merge(hs50_features, on='fid', how='left')
#df_na = df_left.loc[df_left['max0'].isnull()]
#df = df[0:274]
df = df.dropna()

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)

lr = LinearRegression()
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)
test_y['pred_y'] = pred_y
tr_pred_y = lr.predict(train_x)
train_y["pred_y"] = tr_pred_y

from sklearn.linear_model import RidgeCV
train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,2:df.shape[1]], df[['firmness']], test_size=0.25, random_state=42)
train_y = np.log(train_y)
test_y = np.log(test_y)
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_x, train_y)
clf.score(train_x, train_y)
clf_pred_y = clf.predict(test_x)
test_y["pred_y"] = clf_pred_y







