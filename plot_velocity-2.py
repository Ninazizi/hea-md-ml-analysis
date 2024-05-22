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
def cal_pearson(velocity_path,dis_len):
    if 'v00' in io.loadmat(velocity_path).keys():
        velocity = io.loadmat(velocity_path)['v00']
    elif 'v00xy' in io.loadmat(velocity_path).keys():
        velocity = io.loadmat(velocity_path)['v00xy']
    height = velocity[-1, 1, 0] - (2. * velocity[0, 1, 0] - velocity[1, 1, 0])
    gap = velocity[1, 1, 0] - velocity[0, 1, 0]

    print('gap=',gap)
    if gap<10:
        neigh = 20
    elif gap<20:
        neigh = 9
    else:
        neigh = 5
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
    vel_matrix = np.zeros((velocity_value.shape[1]*dis_len,int(2*neigh+1)))
    count = 0
    for ii in range(dis_len,int(dis_len*2)):
        for kk in range(velocity_value.shape[1]):
            vel_matrix[count, 0] = velocity_value[ii,kk]
            for length in range(1,int(neigh+1)):
                vel_matrix[count, length*2-1] = velocity_value[ii - length, kk]
            for length in range(1,int(neigh+1)):
                vel_matrix[count, length*2] = velocity_value[ii + length, kk]
            count+=1
    pearson_list = []
    for ii in range(0,int(2*neigh+1)):
        pearson_list.append(pearsonr(vel_matrix[:,0],vel_matrix[:,ii]).statistic)
        print(pearsonr(vel_matrix[:,0],vel_matrix[:,ii]))
    y=[]
    y.append(pearson_list[0])
    for ii in range(1,int(neigh+1)):
        y.append(0.5*(pearson_list[ii*2-1]+pearson_list[ii*2]))
    x=np.arange(0,int(neigh+1))*gap
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
x_edge1,y_edge1 = cal_pearson(velocity_path,dis_len=34)

data_dir = 'data/sum-edge-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p1_7.mat'
x_edge2,y_edge2 = cal_pearson(velocity_path,dis_len=85)

data_dir = 'data/sum-edge-p1-p2_10112023/'
velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p1_30.mat'
x_edge3,y_edge3 = cal_pearson(velocity_path,dis_len=19)

# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p1.mat'
# x_screw1,y_screw1 = cal_pearson(velocity_path,screw_dis_len)
#
# data_dir = '1011/sum-screw-p1-p2_10112023/'
# velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_p2.mat'
# x_screw2,y_screw2 = cal_pearson(velocity_path,screw_dis_len)

# Function to find the x value where y intersects with Pcc=0.5
def find_intersection_x(x, y):
    x_dense = np.linspace(min(x), max(x), num=1000)  # Create a dense range of x values
    y_dense = np.interp(x_dense, x, y)  # Interpolate the y values on this dense range

    # Find where the interpolated y crosses Pcc=0.5
    idx = np.argwhere(np.diff(np.sign(y_dense - 0.5))).flatten()
    return x_dense[idx]

# Create a single subplot
fig, ax = plt.subplots()#figsize=(3.5, 3.5))
ax.set_ylim([0.205,1.05])
# Plot for the second dataset
ax.scatter(x_edge2, y_edge2, marker='o', c='b', s=10, alpha=0.5, label='9')
ax.plot(x_edge2, y_edge2, c='b', linewidth=1, alpha=0.7)
intersection_x1 = find_intersection_x(x_edge2, y_edge2)
for x in intersection_x1:
    ax.axvline(x=x, c='b', linewidth=1, linestyle='dashed', alpha=0.7, ymax=0.295/0.8 / ax.get_ylim()[1])

# Plot for the first dataset with adjusted marker size and line transparency
ax.scatter(x_edge1, y_edge1, marker='*', c='r', s=30, alpha=0.5, label='15')
ax.plot(x_edge1, y_edge1, c='r', linewidth=1, alpha=0.7)
intersection_x1 = find_intersection_x(x_edge1, y_edge1)
for x in intersection_x1:
    ax.axvline(x=x, c='r', linewidth=1, linestyle='dashed', alpha=0.7, ymax=0.295/0.8 / ax.get_ylim()[1])

# Plot for the third dataset
ax.scatter(x_edge3, y_edge3, marker='^', c='k', s=15, alpha=0.5, label='31')
ax.plot(x_edge3, y_edge3, c='k', linewidth=1, alpha=0.7)
intersection_x1 = find_intersection_x(x_edge3, y_edge3)
for x in intersection_x1:
    ax.axvline(x=x, c='k', linewidth=1, linestyle='dashed', alpha=0.7, ymax=0.295/0.8 / ax.get_ylim()[1])

# Plot the PCC=0.5 line
ax.plot(x_edge1, 0.5*np.ones_like(x_edge1), c='grey', linewidth=1, linestyle='dashed', label='PCC=0.5')

ax.legend(fontsize=13)
plt.tick_params(labelsize=13)

plt.tight_layout()
plt.savefig('fig1-f.jpg')

