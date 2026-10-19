import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#Data augmentation was performed using the B-spline method and the Time-windowing method, where the B-spline method can be regarded as a special case of Time-windowing with a window size of 1.
################################################## max window width = 9
win_width=9
##################################################
#(1)Load spec_data.npy, content_data.npy and time_stamp
rootdir1=r'../../step0_raw_data'
rootdir2=r'../../step2_training_crossval_test_split/time_resolved_spec/plot'
rootdir3=r'../../step2_training_crossval_test_split/time_resolved_spec'
spec_data=np.load(os.path.join(rootdir1,'spec_data.npy'))
content_data=np.load(os.path.join(rootdir1,'content_data.npy'))
tdim_data=np.load(os.path.join(rootdir1,'time_stamp_1.npy'))
M,N1,N2=spec_data.shape[0],spec_data.shape[1],content_data.shape[1]
#(2)Use the first 9 batches as the training set and the last 2 batches as the test set
cali_spec_data_1,test_spec_data_1=spec_data[:M-2],spec_data[M-2:] 
cali_content_data_1,test_content_data_1=content_data[:M-2],content_data[M-2:]
cali_tdim_data_1,test_tdim_data_1=tdim_data[:M-2],tdim_data[M-2:]
#(3)Perform B-spline interpolation on cali_content_data_1
time_series_1=np.arange(4,100,4).tolist() 
time_series_2=np.arange(4,97,2).tolist() 
cali_content_data_2=np.zeros((cali_content_data_1.shape[0],len(time_series_2),1))
for i in range(cali_content_data_1.shape[0]):
    for k in range(1): 
        item_1=cali_content_data_1[i,:,k]
        tck=interpolate.splrep(time_series_1,item_1)
        item_2=interpolate.splev(time_series_2,tck)
        cali_content_data_2[i,:,k]=item_2
        plt.figure()
        plt.plot(time_series_1,item_1)
        plt.plot(time_series_2,item_2)
        plt.savefig(os.path.join(rootdir2,'batch_'+str(i)+'_component_'+str(k)+'.png'))
cali_content_data_2=cali_content_data_2.reshape(cali_content_data_2.shape[0]*cali_content_data_2.shape[1],cali_content_data_2.shape[2])
#(4)Perform interpolation and time-windowing on cali_spec_data_1 so that it corresponds with cali_content_data_2
time_series_3=np.arange(8,193,4).tolist() 
cali_spec_data_2=np.zeros((cali_spec_data_1.shape[0],len(time_series_3),win_width,cali_spec_data_1.shape[2])) 
cali_tdim_data_2=np.zeros((cali_spec_data_1.shape[0],len(time_series_3),win_width,1))
for i in range(cali_spec_data_1.shape[0]):
    for j in range(len(time_series_3)):
        item_1=cali_spec_data_1[i,time_series_3[j]-int((win_width-1)/2):time_series_3[j]+int((win_width+1)/2)] 
        item_2=cali_tdim_data_1[i,time_series_3[j]-int((win_width-1)/2):time_series_3[j]+int((win_width+1)/2)] 
        cali_spec_data_2[i,j,:]=item_1
        cali_tdim_data_2[i,j,:]=item_2
cali_spec_data_2=cali_spec_data_2.reshape(cali_spec_data_2.shape[0]*cali_spec_data_2.shape[1],cali_spec_data_2.shape[2],cali_spec_data_2.shape[3])
cali_tdim_data_2=cali_tdim_data_2.reshape(cali_tdim_data_2.shape[0]*cali_tdim_data_2.shape[1],cali_tdim_data_2.shape[2],cali_tdim_data_2.shape[3])
#(5)B-spline interpolation is not performed on the test set, but the time-windowing method is still used
time_series_4=np.arange(8,193,8).tolist()
test_spec_data_2=np.zeros((test_spec_data_1.shape[0],len(time_series_4),win_width,test_spec_data_1.shape[2])) 
test_tdim_data_2=np.zeros((test_spec_data_1.shape[0],len(time_series_4),win_width,1)) 
for i in range(len(test_spec_data_1)):
    for j in range(len(time_series_4)):
        test_spec_data_2[i,j]=test_spec_data_1[i][time_series_4[j]-int((win_width-1)/2):time_series_4[j]+int((win_width+1)/2)]
        test_tdim_data_2[i,j]=test_tdim_data_1[i][time_series_4[j]-int((win_width-1)/2):time_series_4[j]+int((win_width+1)/2)]
test_content_data_2=test_content_data_1 
#(6)Save Data
np.save(os.path.join(rootdir3,'time_cali_X.npy'),cali_spec_data_2)
np.save(os.path.join(rootdir3,'time_cali_Y.npy'),cali_content_data_2)
np.save(os.path.join(rootdir3,'time_cali_Tdim.npy'),cali_tdim_data_2) 
np.save(os.path.join(rootdir3,'time_test_X.npy'),test_spec_data_2)
np.save(os.path.join(rootdir3,'time_test_Y.npy'),test_content_data_2)  
np.save(os.path.join(rootdir3,'time_test_Tdim.npy'),test_tdim_data_2)                      
