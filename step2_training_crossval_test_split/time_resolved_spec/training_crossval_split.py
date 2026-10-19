import numpy as np
import os
from sklearn.model_selection import KFold
import random
from pathlib import Path

#(1)Load time_cali_X.npy, time_cali_Y.npy, and time_cali_Tdim.npy
rootdir=r'../../step2_training_crossval_test_split/time_resolved_spec'
time_cali_X=np.load(os.path.join(rootdir,'time_cali_X.npy'))
time_cali_Y=np.load(os.path.join(rootdir,'time_cali_Y.npy'))
time_cali_Tdim=np.load(os.path.join(rootdir,'time_cali_Tdim.npy'))
#(2)shuffle the data;otherwise, the deep temporal model will fail
random.seed(42)
idx=random.sample(np.arange(len(time_cali_X)).tolist(),k=len(time_cali_X))
time_cali_X,time_cali_Y,time_cali_Tdim=time_cali_X[idx],time_cali_Y[idx],time_cali_Tdim[idx]
#(3)three-fold cross-validation
name=['fold_1','fold_2','fold_3']
index=0 
kfold=KFold(n_splits=3) 
for train,val in kfold.split(time_cali_X):
    path=Path(os.path.join(rootdir,name[index]))
    path.mkdir(parents=True, exist_ok=True)
    time_train_X,time_train_Y,time_train_Tdim=time_cali_X[train],time_cali_Y[train],time_cali_Tdim[train]
    time_val_X,time_val_Y,time_val_Tdim=time_cali_X[val],time_cali_Y[val],time_cali_Tdim[val]
    np.save(os.path.join(rootdir,name[index],'time_train_X'),time_train_X)
    np.save(os.path.join(rootdir,name[index],'time_train_Y'),time_train_Y)
    np.save(os.path.join(rootdir,name[index],'time_train_Tdim'),time_train_Tdim)
    np.save(os.path.join(rootdir,name[index],'time_val_X'),time_val_X)
    np.save(os.path.join(rootdir,name[index],'time_val_Y'),time_val_Y)
    np.save(os.path.join(rootdir,name[index],'time_val_Tdim'),time_val_Tdim)
    index+=1 