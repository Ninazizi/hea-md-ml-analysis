import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
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
def cal_pearson(csfe_path,gap=17,flag='edge'):
    df = pd.read_csv(csfe_path)
    if flag=='edge':
        df['ave_velocity']=(df['local_velocity']+df['top1_v']+df['top2_v']+df['down1_v']+df['down2_v'])/5.0
    elif flag=='screw':
        df['ave_velocity'] = (df['local_velocity'] + df['top1_v'] + df['top2_v'] +df['top3_v']+ \
                             df['down1_v'] + df['down2_v'] + df['down3_v'])/7.0
    csfe_cols = []
    csfe_cols.append('fe_[0, 1]_mean')
    for i in range(1, 10):
        # csfe_cols.append('fe_['+str((i-0.5)*gap)+', '+str((i+0.5)*gap)+']_mean')
        csfe_cols.append('fe_[' + str(0) + ', ' + str((i) * gap) + ']_mean')
    # plt.hist(df['fe_['+str(0)+', '+str((i)*gap)+']_mean'], color='lightgreen', ec='black', bins=50)
    # plt.title('std=' + str(df['fe_[' + str(0) + ', ' + str((i) * gap) + ']_mean'].std()))
    # plt.show()
    csfe_matrix = df[csfe_cols].values

    gsquare_cols = []
    gsquare_cols.append('gx2+gy2_[0, 1]_mean')
    for i in range(1, 10):
        # gsquare_cols.append('gx2+gy2_[' + str((i - 0.5) * gap) + ', ' + str((i + 0.5) * gap) + ']_mean')
        gsquare_cols.append('gx2+gy2_[' + str(0) + ', ' + str((i) * gap) + ']_mean')
    # plt.hist(df['gx2+gy2_[' + str(0) + ', ' + str((i) * gap) + ']_mean'], color='pink', ec='black', bins=50)
    # plt.title('std='+str(df['gx2+gy2_[' + str(0) + ', ' + str((i) * gap) + ']_mean'].std()))
    # plt.show()
    gsquare_matrix = df[gsquare_cols].values

    print(df.columns)
    vel_cols = ['ave_velocity']
    for i in range(1,10):
        vel_cols.append('top'+str(i)+'_v')
        vel_cols.append('down' + str(i) + '_v')
    vel_matrix = df[vel_cols].values

    nye_cols = [item for item in df.columns if 'nye' in item]
    nye_matrix = df[nye_cols].values

    pearson_list = []
    for ii in range(0,len(nye_cols)):
        pearson_list.append(pearsonr(vel_matrix[:,0],nye_matrix[:,ii]).statistic)
    vel_nye_y = pearson_list
    x=nye_cols
    return x,vel_nye_y


flag = 'edge'
if flag=='edge':
    data_dir = '1011/sum-edge-p1-p2_10112023/'
    # height = edge_box[2]
else:
    data_dir = '1011/sum-screw-p1-p2_10112023/'
    # height = screw_box[2]

data_dir = '1011/'
csfe_path = data_dir + 'edge_p1_final.csv'
x_edge1,y_edge1 = cal_pearson(csfe_path,gap=17,flag='edge')

csfe_path = data_dir + 'edge_p2_final.csv'
x_edge2,y_edge2 = cal_pearson(csfe_path,gap=17,flag='edge')

csfe_path = data_dir + 'screw_p1_final.csv'
x_screw1,y_screw1 = cal_pearson(csfe_path,gap=15,flag='screw')

csfe_path = data_dir + 'screw_p2_final.csv'
x_screw2,y_screw2 = cal_pearson(csfe_path,gap=15,flag='screw')

# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p1.mat'
# x_screw1,y_screw1 = cal_pearson(velocity_path,screw_dis_len)
#
# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p2.mat'
# x_screw2,y_screw2 = cal_pearson(velocity_path,screw_dis_len)

fig,axs = plt.subplots(2,2,figsize=(7, 4),sharex=True,sharey=True)

axs[0,0].set_title('edge leading',weight='bold',size=13)
axs[0,0].bar(np.arange(len(x_edge1)),y_edge1,width=0.8,color=['blue']*4+['orange']*4)

axs[1,1].set_xticks(np.arange(len(x_edge1)), x_edge1)

axs[0,1].set_title('edge trailing',weight='bold',size=13)
axs[0,1].bar(np.arange(len(x_edge2)),y_edge2,width=0.8,color=['blue']*4+['orange']*4)
axs[1,0].set_title('screw leading',weight='bold',size=13)
axs[1,0].bar(np.arange(len(x_screw1)),y_screw1,width=0.8,color=['blue']*4+['orange']*4)
axs[1,1].set_title('screw trailing',weight='bold',size=13)
axs[1,1].bar(np.arange(len(x_screw2)),y_screw2,width=0.8,color=['blue']*4+['orange']*4)
# axs[0,0].legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig12-ave.jpg')

