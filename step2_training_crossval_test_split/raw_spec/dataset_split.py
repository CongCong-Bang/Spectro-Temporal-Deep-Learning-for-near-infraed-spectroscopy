import numpy as np
import os
import random
from sklearn.model_selection import KFold
from pathlib import Path

#(1)Load spec_data.npy and content_data.npy.
rootdir1=r'../../step0_raw_data'
rootdir2=r'../../step2_training_crossval_test_split/raw_spec'
spec_data=np.load(os.path.join(rootdir1,'spec_data.npy'))
content_data=np.load(os.path.join(rootdir1,'content_data.npy'))
M,N=spec_data.shape[0],content_data.shape[1]
#(2)Align spec_data and content_data along the time axis and obtained the aligned time axis
X,Y,Tdim=[],[],[]
for i in range(len(spec_data)):
    temp_1,temp_2=spec_data[i],content_data[i]
    X_idx=np.arange(8,200,8).tolist() 
    X.append(temp_1[X_idx]) 
    Y.append(temp_2) 
    Tdim+=X_idx
X,Y,Tdim=np.array(X),np.array(Y),np.array(Tdim)
X,Y=X.reshape(X.shape[0]*X.shape[1],X.shape[2]),Y.reshape(Y.shape[0]*Y.shape[1],Y.shape[2])
Tdim=Tdim.reshape(X.shape[0],-1)/2 
#(3)Use the first 9 batches as the training set and the last 2 batches as the test set
cali_X,test_X=X[:(M-2)*N],X[(M-2)*N:]
cali_Y,test_Y=Y[:(M-2)*N],Y[(M-2)*N:]
cali_Tdim,test_Tdim=Tdim[:(M-2)*N],Tdim[(M-2)*N:]
#(4)Save data
np.save(os.path.join(rootdir2,'cali_X.npy'),cali_X)
np.save(os.path.join(rootdir2,'cali_Y.npy'),cali_Y)
np.save(os.path.join(rootdir2,'cali_Tdim.npy'),cali_Tdim)
np.save(os.path.join(rootdir2,'test_X.npy'),test_X)
np.save(os.path.join(rootdir2,'test_Y.npy'),test_Y)
np.save(os.path.join(rootdir2,'test_Tdim.npy'),test_Tdim)
#(5)Further apply three-fold cross-validation to cali_X, cali_Y, and cali_Tdim
name=['fold_1','fold_2','fold_3']
index=0 
kfold=KFold(n_splits=3) #three-fold cross-validation
for train,val in kfold.split(cali_X):
    path=Path(os.path.join(rootdir2,name[index]))
    path.mkdir(parents=True, exist_ok=True)
    train_X,train_Y,train_Tdim=cali_X[train],cali_Y[train],cali_Tdim[train]
    val_X,val_Y,val_Tdim=cali_X[val],cali_Y[val],cali_Tdim[val]
    np.save(os.path.join(path,'train_X'),train_X)
    np.save(os.path.join(path,'train_Y'),train_Y)
    np.save(os.path.join(path,'train_Tdim'),train_Tdim)
    np.save(os.path.join(path,'val_X'),val_X)
    np.save(os.path.join(path,'val_Y'),val_Y)
    np.save(os.path.join(path,'val_Tdim'),val_Tdim)
    index+=1 