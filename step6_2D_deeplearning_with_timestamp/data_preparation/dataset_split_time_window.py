import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
import torch
import random
from pathlib import Path

#(1)Load spec_data.npy and content_data.npy.
rootdir1=r'../../step0_raw_data'
rootdir2=r'../../step2_training_crossval_test_split/time_resolved_spec'
rootdir3=r'../../step6_2D_deeplearning_with_timestamp/data_preparation'
spec_data=np.load(os.path.join(rootdir1,'spec_data.npy'))
content_data=np.load(os.path.join(rootdir1,'content_data.npy'))
time_series_data=np.load(os.path.join(rootdir1,'time_stamp_1.npy'))
spec_data=spec_data.reshape(spec_data.shape[0]*spec_data.shape[1],spec_data.shape[2])
time_series_data=time_series_data.reshape(time_series_data.shape[0]*time_series_data.shape[1],1) #将time_series_data按照spec_data的标准进行重构
spec_data=np.hstack((time_series_data,spec_data)) #将time_series_data和spec_data进行合并,成为带时间头的光谱矩阵
content_data=content_data.reshape(content_data.shape[0]*content_data.shape[1],content_data.shape[2])
scaler_1=StandardScaler() #The Z-score standardization method is adopted）
scaler_1.fit(spec_data)
dump(scaler_1,'scaler_1')
#(2)set scaler_2 for Apigenin-7-O-glucuronide
scaler_2=StandardScaler()
scaler_2.fit(content_data[:,0].reshape(-1,1))
dump(scaler_2, 'scaler_2') 

###################### B-spline is a special time windowing method with a window width of 1
t_idx=4
t_window=[[4],np.arange(3,6),np.arange(2,7),np.arange(1,8),np.arange(0,9)]
###############################################################
#(3)Read the three-fold split data and convert it into Torch variables
name=['fold_1','fold_2','fold_3']
for i in range(len(name)):
    path=Path(os.path.join(rootdir3,name[i]))
    path.mkdir(parents=True, exist_ok=True)
    time_train_X=np.load(os.path.join(rootdir2,name[i],'time_train_X.npy'))
    time_train_Y=np.load(os.path.join(rootdir2,name[i],'time_train_Y.npy'))
    time_train_Tdim=np.load(os.path.join(rootdir2,name[i],'time_train_Tdim.npy'))
    time_val_X=np.load(os.path.join(rootdir2,name[i],'time_val_X.npy'))
    time_val_Y=np.load(os.path.join(rootdir2,name[i],'time_val_Y.npy'))
    time_val_Tdim=np.load(os.path.join(rootdir2,name[i],'time_val_Tdim.npy'))
    time_train_X,time_val_X=time_train_X[:,t_window[t_idx],:],time_val_X[:,t_window[t_idx],:]
    time_train_Tdim,time_val_Tdim=time_train_Tdim[:,t_window[t_idx],:],time_val_Tdim[:,t_window[t_idx],:]
    time_train_X=np.concatenate((time_train_Tdim,time_train_X),axis=2)
    time_val_X=np.concatenate((time_val_Tdim,time_val_X),axis=2)
    
    for j in range(time_train_X.shape[0]):
        time_train_X[j]=scaler_1.transform(time_train_X[j])
    for j in range(time_val_X.shape[0]):
        time_val_X[j]=scaler_1.transform(time_val_X[j])
    time_train_Y=scaler_2.transform(time_train_Y)
    time_val_Y=scaler_2.transform(time_val_Y)
    train_X = torch.tensor(time_train_X).float()
    train_Y = torch.tensor(time_train_Y).float()
    val_X = torch.tensor(time_val_X).float()
    val_Y = torch.tensor(time_val_Y).float()
    dump(train_X, os.path.join(rootdir3,name[i],'train_set')) 
    dump(train_Y, os.path.join(rootdir3,name[i],'train_label')) 
    dump(val_X, os.path.join(rootdir3,name[i],'val_set')) 
    dump(val_Y, os.path.join(rootdir3,name[i],'val_label'))
#(4)Read the test data and convert it into Torch variables
time_test_X=np.load(os.path.join(rootdir2,'time_test_X.npy'))
time_test_Y=np.load(os.path.join(rootdir2,'time_test_Y.npy'))
time_test_Tdim=np.load(os.path.join(rootdir2,'time_test_Tdim.npy'))
time_test_X=time_test_X.reshape(time_test_X.shape[0]*time_test_X.shape[1],time_test_X.shape[2],time_test_X.shape[3])
time_test_Y=time_test_Y.reshape(time_test_Y.shape[0]*time_test_Y.shape[1],time_test_Y.shape[2])
time_test_Tdim=time_test_Tdim.reshape(time_test_Tdim.shape[0]*time_test_Tdim.shape[1],time_test_Tdim.shape[2],1)
time_test_X=time_test_X[:,t_window[t_idx],:]
time_test_Tdim=time_test_Tdim[:,t_window[t_idx],:]
time_test_X=np.concatenate((time_test_Tdim,time_test_X),axis=2)
for i in range(time_test_X.shape[0]):
    time_test_X[i]=scaler_1.transform(time_test_X[i])
time_test_Y=scaler_2.transform(time_test_Y)
test_X = torch.tensor(time_test_X).float()
test_Y = torch.tensor(time_test_Y).float()
dump(test_X, os.path.join(rootdir3,'test_set')) 
dump(test_Y, os.path.join(rootdir3,'test_label')) 