import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
def cal_pearson(csfe_path,gap=17):
    df = pd.read_csv(csfe_path)

    print(df.columns)
    vel_cols = ['local_velocity']
    for i in range(1,10):
        vel_cols.append('top'+str(i)+'_v')
        vel_cols.append('down' + str(i) + '_v')
    vel_matrix = df[vel_cols].values

    gsquare_cols = []
    gsquare_cols.append('gx2+gy2_[0, 1]_mean')
    for i in range(1, 10):
        # gsquare_cols.append('gx2+gy2_[' + str((i - 0.5) * gap) + ', ' + str((i + 0.5) * gap) + ']_mean')
        gsquare_cols.append('gx2+gy2_[' + str(0) + ', ' + str((i) * gap) + ']_mean')
    gsquare_matrix = df[gsquare_cols].values
    print('gsquare=',np.mean(gsquare_matrix[:,3]))

    pearson_list = []
    for ii in range(0, 10):
        pearson_list.append(pearsonr(gsquare_matrix[:, 0], gsquare_matrix[:, ii]).statistic)
        print(pearsonr(gsquare_matrix[:, 0], gsquare_matrix[:, ii]))
    gsquare_y = pearson_list

    x=np.arange(0,10)*gap
    return x,np.array(gsquare_y)


flag = 'edge'
if flag=='edge':
    data_dir = 'data/sum-edge-p1-p2_10112023/'
    # height = edge_box[2]
else:
    data_dir = 'data/sum-screw-p1-p2_10112023/'
    # height = screw_box[2]

data_dir = 'data/'
csfe_path = data_dir + 'edge_p1_final.csv'
x_edge1,y_edge1 = cal_pearson(csfe_path,gap=17)

csfe_path = data_dir + 'edge_p2_final.csv'
x_edge2,y_edge2 = cal_pearson(csfe_path,gap=17)

csfe_path = data_dir + 'screw_p1_final.csv'
x_screw1,y_screw1 = cal_pearson(csfe_path,gap=15)

csfe_path = data_dir + 'screw_p2_final.csv'
x_screw2,y_screw2 = cal_pearson(csfe_path,gap=15)

# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p1.mat'
# x_screw1,y_screw1 = cal_pearson(velocity_path,screw_dis_len)
#
# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p2.mat'
# x_screw2,y_screw2 = cal_pearson(velocity_path,screw_dis_len)

fig,axs = plt.subplots(2,2,figsize=(7, 7),sharex=True,sharey=True)

axs[0,0].scatter(x_edge1,y_edge1,marker='*', c='r',s=38,alpha=0.8,label='leading_edge')
axs[0,0].plot(x_edge1,y_edge1,c='r',linewidth=1)
axs[0,0].plot(x_edge1,0.4*np.ones_like(x_edge1),c='b',linewidth=1,linestyle='dashed', label='PCC=0.4')
axs[0,0].plot(x_edge1,0.5*np.ones_like(x_edge1),c='grey',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[0,0].plot(x_edge1,-0.2*np.ones_like(x_edge1),c='b',linewidth=1,linestyle='dashed', label='PCC=-0.2')
# axs[0,0].set_xticks(x)
axs[0,0].legend()
axs[0,1].scatter(x_edge2,y_edge2,marker='*', c='k',s=38,alpha=0.8,label='trailing_edge')
axs[0,1].plot(x_edge2,y_edge2,c='k',linewidth=1)
axs[0,1].plot(x_edge2,0.4*np.ones_like(x_edge2),c='b',linewidth=1,linestyle='dashed', label='PCC=0.4')
axs[0,1].plot(x_edge2,0.5*np.ones_like(x_edge2),c='grey',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[0,1].plot(x_edge2,-0.2*np.ones_like(x_edge2),c='b',linewidth=1,linestyle='dashed', label='PCC=-0.2')
# axs[0,1].set_xticks(x)
axs[0,1].legend()
axs[1,0].scatter(x_screw1,y_screw1,marker='o', c='r',s=35,alpha=0.8,label='leading_screw')
axs[1,0].plot(x_screw1,y_screw1,c='r',linewidth=1)
axs[1,0].plot(x_screw1,0.4*np.ones_like(x_screw1),c='b',linewidth=1,linestyle='dashed', label='PCC=0.4')
axs[1,0].plot(x_screw1,0.5*np.ones_like(x_screw1),c='grey',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[1,0].plot(x_screw1,-0.2*np.ones_like(x_screw1),c='b',linewidth=1,linestyle='dashed', label='PCC=-0.2')
# axs[1,0].set_xticks(x)
axs[1,0].legend()
axs[1,1].scatter(x_screw2,y_screw2,marker='o', c='k',s=35,alpha=0.8,label='trailing_screw')
axs[1,1].plot(x_screw2,0.4*np.ones_like(x_screw2),c='b',linewidth=1,linestyle='dashed', label='PCC=0.4')
axs[1,1].plot(x_screw2,0.5*np.ones_like(x_screw2),c='grey',linewidth=1,linestyle='dashed', label='PCC=0.5')
# axs[1,1].plot(x_screw2,-0.2*np.ones_like(x_screw2),c='b',linewidth=1,linestyle='dashed', label='PCC=-0.2')
axs[1,1].plot(x_screw2,y_screw2,c='k',linewidth=1)
# axs[1,1].set_xticks(x)
axs[1,1].legend()

plt.tight_layout()
plt.savefig('fig10.jpg')

