import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#根Based on the given training set and validation set, select the optimal number of PLS latent variables
def opt_pls(X_train,Y_train,X_val,Y_val,n_components):
    if X_train.shape[1]<len(n_components):
        n_components=np.arange(1,X_train.shape[1])
    index,r2_npy=0,np.zeros(len(n_components))
    for item in n_components:
        pls=PLSRegression(n_components=item)
        pls.fit(X_train,Y_train)
        val_pred=pls.predict(X_val).flatten()
        val_r2=r2_score(Y_val,val_pred)
        r2_npy[index]=val_r2
        index+=1
    opt_components=n_components[np.argmax(r2_npy)]
    return opt_components

rootdir1=r'../step2_training_crossval_test_split/raw_spec'
cali_X,test_X=np.load(os.path.join(rootdir1,'cali_X.npy')),np.load(os.path.join(rootdir1,'test_X.npy'))
cali_Y,test_Y=np.load(os.path.join(rootdir1,'cali_Y.npy')),np.load(os.path.join(rootdir1,'test_Y.npy'))

#Three-fold corss-validation
#1:represents 1 bioactive compound;3:3-fold cross-validation;7:the number of latent variables,rmse_c,r2_c,rmse_cv,r2_cv,rmse_t,r2_t
sta=np.zeros((1,3,8)) 
n_components=np.arange(1,30)
for i in range(1):
    index,temp_sta=0,np.zeros((3,8)) 
    kfold=KFold(n_splits=3) 
    cali_X_copy,test_X_copy=cali_X.copy(),test_X.copy()
    cali_Y_copy,test_Y_copy=cali_Y.copy(),test_Y.copy()
    cali_Y_copy,test_Y_copy=cali_Y_copy[:,i],test_Y_copy[:,i]
    test_pred_list=[] 
    for train,val in kfold.split(cali_X_copy):
        opt_components=opt_pls(cali_X_copy[train],cali_Y_copy[train],cali_X_copy[val],cali_Y_copy[val],n_components)
        #Fit the model based on the training set results
        pls=PLSRegression(n_components=opt_components)
        pls.fit(cali_X_copy[train],cali_Y_copy[train])
        #Use the PLS model to predict the training set
        train_pred=pls.predict(cali_X_copy[train]).flatten()
        train_r2=r2_score(cali_Y_copy[train],train_pred) 
        train_rmse=mean_squared_error(cali_Y_copy[train],train_pred,multioutput='uniform_average')**0.5
        #Use the PLS model to predict the cross-validation set
        val_pred=pls.predict(cali_X_copy[val]).flatten()
        val_r2=r2_score(cali_Y_copy[val],val_pred) 
        val_rmse=mean_squared_error(cali_Y_copy[val],val_pred,multioutput='uniform_average')**0.5
        #Use the PLS model to predict the test set
        test_pred=pls.predict(test_X_copy).flatten()
        test_r2=r2_score(test_Y_copy,test_pred) 
        test_rmse=mean_squared_error(test_Y_copy,test_pred,multioutput='uniform_average')**0.5
        test_pred_list.append(test_pred)
        test_rpd=np.std(test_Y_copy)/test_rmse
        #Write the data to temp_sta
        temp_sta[index,:]=np.array([opt_components,train_r2,train_rmse,val_r2,val_rmse,test_r2,test_rmse,test_rpd])
        index+=1
    mean_test_pred=np.mean(np.array(test_pred_list),axis=0)
    #Plot the curve of the true values (test_Y_copy) and the curve of the predicted values (mean_test_pred), and compare them
    time_series=np.arange(8,200,8).tolist()
    plt.figure() 
    plt.plot(time_series,test_Y_copy[:24],color='red')
    plt.plot(time_series,mean_test_pred[:24],color='black')
    plt.savefig('compound_'+str(i)+'_test_batch_1.png')
    plt.figure() 
    plt.plot(time_series,test_Y_copy[24:],color='red')
    plt.plot(time_series,mean_test_pred[24:],color='black')
    plt.savefig('compound_'+str(i)+'_test_batch_2.png')
    #Write the temp_sta to sta
    sta[i]=temp_sta
aver_sta=np.mean(sta,axis=1)