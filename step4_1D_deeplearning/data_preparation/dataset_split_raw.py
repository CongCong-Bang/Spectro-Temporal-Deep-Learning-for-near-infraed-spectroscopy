import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
import torch
import random
from pathlib import Path

#(1)Load spec_data.npy and content_data.npy
rootdir1=r'../../step0_raw_data'
rootdir2=r'../../step2_training_crossval_test_split/raw_spec'
rootdir3=r'../../step4_1D_deeplearning/data_preparation'
spec_data=np.load(os.path.join(rootdir1,'spec_data.npy'))
content_data=np.load(os.path.join(rootdir1,'content_data.npy'))
spec_data=spec_data.reshape(spec_data.shape[0]*spec_data.shape[1],spec_data.shape[2])
content_data=content_data.reshape(content_data.shape[0]*content_data.shape[1],content_data.shape[2])
scaler_1=StandardScaler()#The Z-score standardization method is adopted
scaler_1.fit(spec_data)
dump(scaler_1,'scaler_1')
#(2)set scaler_2 for Apigenin-7-O-glucuronide
scaler_2=StandardScaler()
scaler_2.fit(content_data[:,0].reshape(-1,1))
dump(scaler_2, 'scaler_2') 
#(3)Read the three-fold split data and convert it into Torch variables
name=['fold_1','fold_2','fold_3']
for i in range(len(name)):
    path=Path(os.path.join(rootdir3,name[i]))
    path.mkdir(parents=True, exist_ok=True)
    train_X=np.load(os.path.join(rootdir2,name[i],'train_X.npy'))
    train_Y=np.load(os.path.join(rootdir2,name[i],'train_Y.npy'))
    val_X=np.load(os.path.join(rootdir2,name[i],'val_X.npy'))
    val_Y=np.load(os.path.join(rootdir2,name[i],'val_Y.npy'))
    train_X,val_X=train_X[:,np.newaxis,:],val_X[:,np.newaxis,:]
    for j in range(train_X.shape[0]):
        train_X[j]=scaler_1.transform(train_X[j])
    for j in range(val_X.shape[0]):
        val_X[j]=scaler_1.transform(val_X[j])
    train_Y=scaler_2.transform(train_Y)
    val_Y=scaler_2.transform(val_Y)
    train_X = torch.tensor(train_X).float()
    train_Y = torch.tensor(train_Y).float()
    val_X = torch.tensor(val_X).float()
    val_Y = torch.tensor(val_Y).float()
    dump(train_X, os.path.join(rootdir3,name[i],'train_set')) 
    dump(train_Y, os.path.join(rootdir3,name[i],'train_label')) 
    dump(val_X, os.path.join(rootdir3,name[i],'val_set')) 
    dump(val_Y, os.path.join(rootdir3,name[i],'val_label'))
#(4)Read the test data and convert it into Torch variables
test_X=np.load(os.path.join(rootdir2,'test_X.npy'))
test_Y=np.load(os.path.join(rootdir2,'test_Y.npy'))
test_X=test_X[:,np.newaxis,:]
for i in range(test_X.shape[0]):
    test_X[i]=scaler_1.transform(test_X[i])
test_Y=scaler_2.transform(test_Y)
test_X = torch.tensor(test_X).float()
test_Y = torch.tensor(test_Y).float()
dump(test_X, os.path.join(rootdir3,'test_set')) 
dump(test_Y, os.path.join(rootdir3,'test_label')) 