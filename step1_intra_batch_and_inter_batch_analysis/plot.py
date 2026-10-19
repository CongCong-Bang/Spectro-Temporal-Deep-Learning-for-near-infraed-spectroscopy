import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
from sklearn.decomposition import PCA
import scipy.signal as signal

matplotlib.rcParams['font.sans-serif'] = ['SimHei']          # 黑体
matplotlib.rcParams['font.sans-serif'] = ['SimHei',            # 优先用黑体
                                          'SimSun',             # 备用：宋体
                                          'Times New Roman',        # 备用：英文
                                          ]       
matplotlib.rcParams['axes.unicode_minus'] = False              # 正常显示负号（重要！）

def deriv2(X,WINDOW=11,ORDER=3,DERIV=2):
    return signal.savgol_filter(X,WINDOW,ORDER,DERIV)

#Take training batch tr1 as an example
rootdir=r'../step0_raw_data'
all_spec=np.load(os.path.join(rootdir,'spec_data.npy'))
bands=np.load(os.path.join(rootdir,'band.npy'))
all_content=np.load(os.path.join(rootdir,'content_data.npy'))
spec=all_spec[0]
content=all_content[0]
#Sequentially plot the spectral evolution,the second-order derivative spectra in 1330-1660 nm range,
#the charomatographic process trajectory of apigenin-7-O-glucuronide, and the overall spectral evolution trend line
#(*1)the spectral evolution
hot=plt.cm.get_cmap('coolwarm')
plt.figure(figsize=(8,3))
plt.rc('font',family='Times New Roman') 
plt.tick_params(labelsize=14)
plt.ylim(-0.5,2.1)
plt.yticks(np.arange(-0.5,2.1,0.5))
for j in range(len(spec)):
    temp=spec[j]
    plt.plot(bands,temp,c=hot(j/201),linewidth=0.5) #共采集100分钟,共有201条光谱
plt.savefig('the spectral evolution.png',dpi=300)
#(*2)the second-order derivative spectra in 1330-1660 nm range
deriv2_spec=deriv2(spec) 
deriv2_spec=deriv2_spec[:,280:388]
deriv2_band=bands[280:388]
hot=plt.cm.get_cmap('coolwarm')
plt.figure(figsize=(5,5))
plt.rc('font',family='Times New Roman') 
plt.xlim(1300,1660)
plt.xticks(np.arange(1300,1660,80))
plt.ylim(-7.8*0.001,5.1*0.001)
plt.yticks(np.arange(-7.5*0.001,5.1*0.001,2.5*0.001))
plt.tick_params(labelsize=24)
ax=plt.gca() 
ax.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
ax.yaxis.get_offset_text().set_fontsize(24) 
for j in range(len(deriv2_spec)):
    temp=deriv2_spec[j]
    plt.plot(deriv2_band,temp,c=hot(j/201),linewidth=0.5)
plt.savefig('the second-order derivative spectra in 1330-1660.png',dpi=300)
#(*3)the charomatographic process trajectory of apigenin-7-O-glucuronide
time_stamp=np.arange(4,100,4).tolist() 
time_stamp=np.array(time_stamp)
fig,ax=plt.subplots(figsize=(4,2))
plt.rc('font',family='Times New Roman') 
ax.plot(time_stamp,content[:,0],c='#66B2FF',label='Apigenin 7-O-glucoside') 
plt.legend(loc='best',fontsize=5)
plt.savefig('the charomatographic process trajectory.png',dpi=300)
#(*4)the overall spectral evolution trend line
X_idx=np.arange(0,201,8).tolist() 
time_stamp=np.arange(0,100.5,4) # 光谱和含量变化共用的时间戳
trans_spec=np.zeros((len(X_idx),spec.shape[1]))
trans_spec=spec[X_idx,:]
trend_data=np.zeros(trans_spec.shape[0])
lower_ref=np.mean(trans_spec[:,:280]) #Use the average transmitted light intensity over the spectral range of 833–1300 nm as the lower reference limit
trend_data=(trans_spec[:,333]+trans_spec[:,444]-2*lower_ref)/(trans_spec[:,322]+trans_spec[:,441]-2*lower_ref)
fig,ax=plt.subplots(figsize=(4,2))
plt.xlim(0,101)
plt.ylim(0.88,1.19)
plt.xticks(np.arange(0,101,10))
plt.yticks(np.arange(0.88,1.19,0.05))
plt.rc('font',family='Times New Roman') 
plt.tick_params(labelsize=12)
ax.plot(time_stamp,trend_data)
ax.xaxis.tick_top()
ax.invert_yaxis()
plt.savefig('the overall spectral evolution trend line.png') 