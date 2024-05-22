import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
import copy
from scipy.stats import pearsonr

pd.set_option('display.max_columns', None)
font = {
    'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'bold',
    # 'weight': 'normal',
    'size': 10,
}
plt.rc('font', **font)
plt.rcParams['figure.dpi'] = 600
def cal_pearson(velocity_path,dis_len):
    velocity = io.loadmat(velocity_path)['v00xy']
    height = velocity[-1,1,0]-(2.*velocity[0,1,0]-velocity[1,1,0])
    gap = velocity[1,1,0]-velocity[0,1,0]
    print('gap=',gap)
    print(velocity.shape)
    # create the periodic top and bottom for csfe, strain and stress
    top_velocity = copy.deepcopy(velocity)
    top_velocity[:,1,:]+=height
    bottom_velocity = copy.deepcopy(velocity)
    bottom_velocity[:,1,:]-=height
    velocity1=np.concatenate([bottom_velocity,
                          velocity,
                          top_velocity])
    velocity = velocity1
    velocity_value = velocity[:,2,:]
    print(velocity.shape)
    vel_matrix = np.zeros((velocity_value.shape[1]*dis_len,19))
    count = 0
    for ii in range(dis_len,int(dis_len*2)):
        for kk in range(velocity_value.shape[1]):
            vel_matrix[count, 0] = velocity_value[ii,kk]
            for length in range(1,10):
                vel_matrix[count, length*2-1] = velocity_value[ii - length, kk]
            for length in range(1,10):
                vel_matrix[count, length*2] = velocity_value[ii + length, kk]
            count+=1
    pearson_list = []
    for ii in range(0,19):
        pearson_list.append(pearsonr(vel_matrix[:,0],vel_matrix[:,ii]).statistic)
        print(pearsonr(vel_matrix[:,0],vel_matrix[:,ii]))
    y=[]
    y.append(pearson_list[0])
    for ii in range(1,10):
        y.append(0.5*(pearson_list[ii*2-1]+pearson_list[ii*2]))
    x=np.arange(0,10)*gap
    return x,y

edge_dis_len=34
screw_dis_len=45
flag = 'edge'
if flag=='edge':
    data_dir = 'data/sum-edge-p1-p2_10112023/'
    # height = edge_box[2]
else:
    data_dir = 'data/sum-screw-p1-p2_10112023/'
    # height = screw_box[2]

data_dir = 'data/sum-edge-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p1.mat'
x_edge1,y_edge1 = cal_pearson(velocity_path,edge_dis_len)

data_dir = 'data/sum-edge-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p2.mat'
x_edge2,y_edge2 = cal_pearson(velocity_path,edge_dis_len)

data_dir = 'data/sum-screw-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p1.mat'
x_screw1,y_screw1 = cal_pearson(velocity_path,screw_dis_len)

data_dir = 'data/sum-screw-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p2.mat'
x_screw2,y_screw2 = cal_pearson(velocity_path,screw_dis_len)

fig,axs = plt.subplots(2,2,figsize=(7, 7),sharex=True,sharey=True)

axs[0,0].scatter(x_edge1,y_edge1,marker='*', c='r',s=38,alpha=0.8,label='leading_edge')
axs[0,0].plot(x_edge1,y_edge1,c='r',linewidth=1)
axs[0,0].plot(x_edge1,0.5*np.ones_like(x_edge1),c='b',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[0,0].set_xticks(x)
axs[0,0].legend()
axs[0,1].scatter(x_edge2,y_edge2,marker='*', c='k',s=38,alpha=0.8,label='trailing_edge')
axs[0,1].plot(x_edge2,y_edge2,c='k',linewidth=1)
axs[0,1].plot(x_edge2,0.5*np.ones_like(x_edge2),c='b',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[0,1].set_xticks(x)
axs[0,1].legend()
axs[1,0].scatter(x_screw1,y_screw1,marker='o', c='r',s=35,alpha=0.8,label='leading_screw')
axs[1,0].plot(x_screw1,y_screw1,c='r',linewidth=1)
axs[1,0].plot(x_screw1,0.5*np.ones_like(x_screw1),c='b',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[1,0].set_xticks(x)
axs[1,0].legend()
axs[1,1].scatter(x_screw2,y_screw2,marker='o', c='k',s=35,alpha=0.8,label='trailing_screw')
axs[1,1].plot(x_screw2,0.5*np.ones_like(x_screw2),c='b',linewidth=1,linestyle='dashed', label='PCC=0.5')
axs[1,1].plot(x_screw2,y_screw2,c='k',linewidth=1)
# axs[1,1].set_xticks(x)
axs[1,1].legend()

plt.tight_layout()
plt.savefig('fig8.jpg')

