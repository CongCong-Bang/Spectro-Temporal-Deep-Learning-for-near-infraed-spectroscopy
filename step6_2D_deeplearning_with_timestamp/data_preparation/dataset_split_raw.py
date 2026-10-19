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
rootdir3=r'../../step6_2D_deeplearning_with_timestamp/data_preparation'
spec_data=np.load(os.path.join(rootdir1,'spec_data.npy'))
content_data=np.load(os.path.join(rootdir1,'content_data.npy'))
time_series_data=np.load(os.path.join(rootdir1,'time_stamp_1.npy'))
spec_data=spec_data.reshape(spec_data.shape[0]*spec_data.shape[1],spec_data.shape[2])
time_series_data=time_series_data.reshape(time_series_data.shape[0]*time_series_data.shape[1],1) #将time_series_data按照spec_data的标准进行重构
spec_data=np.hstack((time_series_data,spec_data)) #将time_series_data和spec_data进行合并,成为带时间头的光谱矩阵
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
    train_Tdim=np.load(os.path.join(rootdir2,name[i],'train_Tdim.npy'))
    val_X=np.load(os.path.join(rootdir2,name[i],'val_X.npy'))
    val_Y=np.load(os.path.join(rootdir2,name[i],'val_Y.npy'))
    val_Tdim=np.load(os.path.join(rootdir2,name[i],'val_Tdim.npy'))
    train_Tdim,val_Tdim=train_Tdim[:,np.newaxis,:],val_Tdim[:,np.newaxis,:]
    train_X,val_X=train_X[:,np.newaxis,:],val_X[:,np.newaxis,:]
    train_X=np.concatenate((train_Tdim,train_X),axis=2) #合并后光谱维变成508+1,其中1是时间戳维度
    val_X=np.concatenate((val_Tdim,val_X),axis=2)
    
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
test_Tdim=np.load(os.path.join(rootdir2,'test_Tdim.npy'))
test_X,test_Tdim=test_X[:,np.newaxis,:],test_Tdim[:,np.newaxis,:]
test_X=np.concatenate((test_Tdim,test_X),axis=2)
for i in range(test_X.shape[0]):
    test_X[i]=scaler_1.transform(test_X[i])
test_Y=scaler_2.transform(test_Y)
test_X = torch.tensor(test_X).float()
test_Y = torch.tensor(test_Y).float()
dump(test_X, os.path.join(rootdir3,'test_set')) 
dump(test_Y, os.path.join(rootdir3,'test_label')) 