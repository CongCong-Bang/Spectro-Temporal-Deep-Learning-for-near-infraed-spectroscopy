import matplotlib.pyplot as plt
import numpy as np
import os

#(1)Read the SHAP values and band information.
rootdir1=r'../../step0_raw_data'
shap_values=np.load('mean_shap_values.npy')
shap_values=np.abs(shap_values)
band=np.load(os.path.join(rootdir1,'band.npy'))
band=[int(band[val]) for val in np.arange(0,len(band),20)]

#(2)Plot the SHAP values of the time series spectra for testing batch b1 and the independent SHAP values of the time series.
#(2.1)Spectral dimension
b1_shap_values=np.mean(shap_values[:24,:,1:],axis=1) #Average along the time dimension
b1_shap_values=np.expand_dims(b1_shap_values,1) 
b1_shap_values=b1_shap_values.repeat(5,axis=1) #To make the display more visually appealing
plt.figure()
for i in range(len(b1_shap_values)):
    plt.subplot(24,1,i+1)
    plt.subplots_adjust(wspace=0, hspace=-0.98)  
    temp=(b1_shap_values[i]-np.min(b1_shap_values[i]))/(np.max(b1_shap_values[i])-np.min(b1_shap_values[i]))
    plt.imshow(temp, cmap=plt.cm.hot, vmin=0, vmax=1)
    if i==len(b1_shap_values)-1:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks([])
        plt.yticks([])
plt.savefig('spectral_shap_1.png')
#(2.2)Temporal dimension
b1_shap_time=np.mean(shap_values[:24],axis=2)
plt.figure(figsize=(8,3.5))
hot=plt.cm.get_cmap('coolwarm')
plt.rc('font',family='Times New Roman') 
plt.xticks([])
plt.tick_params(labelsize=32)
plt.subplots_adjust(bottom=0.15)
ax=plt.gca() #Set it to scientific notation and adjust the font size
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.get_offset_text().set(size=32)  
for i in range(len(b1_shap_time)):
    plt.plot(np.arange(1,b1_shap_time.shape[1]+1),b1_shap_time[i],c=hot(i/len(b1_shap_time)),linewidth=1)
plt.savefig('time_shap_1.png')

#(3)Plot the SHAP values of the time series spectra for testing batch b2 and the independent SHAP values of the time series.
#(3.1)Spectral dimension
b2_shap_values=np.mean(shap_values[24:,:,1:],axis=1) #Average along the spectral dimension
b2_shap_values=np.expand_dims(b2_shap_values,1) 
b2_shap_values=b2_shap_values.repeat(5,axis=1) #To make the display more visually appealing
plt.figure()
for i in range(len(b2_shap_values)):
    plt.subplot(24,1,i+1)
    plt.subplots_adjust(wspace=0, hspace=-0.98)  
    temp=(b2_shap_values[i]-np.min(b2_shap_values[i]))/(np.max(b2_shap_values[i])-np.min(b2_shap_values[i]))
    plt.imshow(temp, cmap=plt.cm.hot, vmin=0, vmax=1)
    if i==len(b2_shap_values)-1:
        plt.xticks(ticks=np.arange(0, 508, 20), labels=[f"{i}" for i in band], rotation=90, fontsize=13)
        plt.yticks([])
    else:
        plt.xticks([])
        plt.yticks([])
plt.savefig('spectral_shap_2.png')
#(3.2)Temporal dimension
b2_shap_time=np.mean(shap_values[24:],axis=2)
plt.figure(figsize=(8,3.5))
hot=plt.cm.get_cmap('coolwarm')
plt.rc('font',family='Times New Roman') 
plt.xticks(np.arange(1,10))
plt.tick_params(labelsize=32)
plt.subplots_adjust(bottom=0.15)
ax=plt.gca() #Set it to scientific notation and adjust the font size
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.get_offset_text().set(size=32)  
for i in range(len(b2_shap_time)):
    plt.plot(np.arange(1,b2_shap_time.shape[1]+1),b2_shap_time[i],c=hot(i/len(b2_shap_time)),linewidth=1)
plt.savefig('time_shap_2.png')