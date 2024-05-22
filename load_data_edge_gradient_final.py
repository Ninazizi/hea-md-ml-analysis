from scipy import io
import numpy as np

import pandas as pd
import copy

def calculate_gradient(fe,axis=0):
    newfe = fe.reshape((-1,33,3))
    grid_fe = newfe[:,:,2]
    grad_csfe_value = np.gradient(grid_fe,axis=axis)
    grad_csfe = np.zeros_like(newfe)
    if axis==0:
        grad_csfe[:,:,2] = grad_csfe_value/15.09
    else:
        grad_csfe[:, :, 2] = grad_csfe_value/17.43
    grad_csfe[:, :, 0] = newfe[:,:,0]
    grad_csfe[:, :, 1] = newfe[:, :, 1]
    grad_csfe = grad_csfe.reshape((-1,3))
    return grad_csfe

def find_match_csfe_nye(nye, local_pos_vel,csfe,best_match=[0,1]):
    distance_arr = np.linalg.norm((csfe[:,:2] - (local_pos_vel[:2]).reshape(1,-1)),axis=1)
    x_distance = csfe[:,0]-local_pos_vel[0]
    if best_match[1]==1:
        min_distance_index = np.argmin(distance_arr)

        return csfe[min_distance_index,2], None
    elif best_match[1]>1:
        # min_distance_indexes = np.argsort(distance_arr)[:best_match]
        # print(best_match,min(distance_arr))
        min_distance_indexes = (distance_arr<best_match[1])
        min_distance_indexes2 = (distance_arr>=best_match[0])
        min_distance_indexes = min_distance_indexes*min_distance_indexes2
        if np.sum(min_distance_indexes)<1:
            print('blank')

        return csfe[min_distance_indexes,2], None

def adjust_index(value,maxvalue):
    if value < 0:
        value += maxvalue
    elif value >= maxvalue:
        value -= maxvalue
    return value

def sfe_feature(nye, velocity, csfe, best_match=[0,17], average_version='mean'):
    velocity_with_localcsfe = np.full([velocity.shape[0],49,velocity.shape[2]], np.nan)
    # velocity_with_localcsfe = np.zeros((velocity.shape[0],14,velocity.shape[2]))

    # for n_segment in range(9,velocity.shape[0]-9):
    for n_segment in range(velocity.shape[0]):
        for n_dis in range(nye.shape[0]):
            local_pos_velocity = velocity[n_segment,:,n_dis]
            indexes = [value for value in range(n_segment-2,n_segment+3)]
            for i in range(len(indexes)):
                if indexes[i]<0:
                    indexes[i]+=len(indexes)
                elif indexes[i]>=len(indexes):
                    indexes[i]-=len(indexes)

            local_nye_max = np.nanmax(nye[n_dis, indexes])
            local_nye_min = np.nanmin(nye[n_dis, indexes])
            local_nye_mean = np.nanmean(nye[n_dis, indexes])
            local_nye = nye[n_dis, n_segment]

            if np.isnan(local_nye_min) or np.isnan(local_nye_max):
                # pass
                velocity_with_localcsfe[n_segment, 3, n_dis] = np.nan
            # elif local_pos_velocity[0]<(15*45):
            #     velocity_with_localcsfe[n_segment, 3, n_dis] = np.nan
            # elif local_pos_velocity[0]>(1019-(15*45)):
            #     velocity_with_localcsfe[n_segment, 3, n_dis] = np.nan
            else:
                local_csfe, _ = find_match_csfe_nye(nye, local_pos_velocity,csfe,best_match=best_match)
                # dis_number = n_dis
                velocity_with_localcsfe[n_segment, 0, n_dis] = local_pos_velocity[0]
                velocity_with_localcsfe[n_segment, 1, n_dis] = local_pos_velocity[1]
                velocity_with_localcsfe[n_segment, 2, n_dis] = local_pos_velocity[2]

                velocity_with_localcsfe[n_segment, 3, n_dis] = local_nye_max
                # velocity_with_localcsfe[n_segment, 4, n_dis] = np.mean(local_csfe)
                # velocity_with_localcsfe[n_segment, 5, n_dis] = np.std(local_csfe)
                if best_match[1]>1:
                    count_within_cutoff = len(local_csfe)
                else:
                    count_within_cutoff = 1
                    local_csfe = np.array(local_csfe)
                velocity_with_localcsfe[n_segment, 27, n_dis] = np.mean(local_csfe)
                velocity_with_localcsfe[n_segment, 28, n_dis] = np.std(local_csfe)
                velocity_with_localcsfe[n_segment, 29, n_dis] = np.mean(local_csfe**2.0)
                # velocity_with_localcsfe[n_segment, 29, n_dis] = np.mean(np.abs(local_csfe))
                # else:
                #     velocity_with_localcsfe[n_segment, 4, n_dis] = local_csfe

                for delta in range(1,10):
                    velocity_with_localcsfe[n_segment, 4+delta*2, n_dis] = velocity[adjust_index(n_segment+delta,velocity.shape[0]), 2, n_dis]
                    velocity_with_localcsfe[n_segment, 5+delta*2, n_dis] = velocity[adjust_index(n_segment-delta,velocity.shape[0]), 2, n_dis]
                velocity_with_localcsfe[n_segment, 24, n_dis] = local_nye_min
                velocity_with_localcsfe[n_segment, 25, n_dis] = local_nye_mean
                velocity_with_localcsfe[n_segment, 26, n_dis] = local_nye

    return velocity_with_localcsfe

def construct_feature(df,nye,velocity,csfe,
                      gradient_csfe_x,
                      gradient_csfe_y,
                      cutoff=[0,30]):
    velocity_with_localcsfe = sfe_feature(nye, velocity,csfe,best_match=cutoff)
    local_csfe_9_mean = (velocity_with_localcsfe[:, 27, :]).reshape(-1)
    df['fe_' + str(cutoff) + '_mean'] = local_csfe_9_mean

    gradient_csfe_square = np.zeros_like(gradient_csfe_x)
    gradient_csfe_square[:,:2] = gradient_csfe_x[:,:2]
    gradient_csfe_square[:,2] = gradient_csfe_x[:,2]**2.0+gradient_csfe_y[:,2]**2.0
    velocity_with_localcsfe = sfe_feature(nye, velocity, gradient_csfe_square, best_match=cutoff)
    local_csfe_9_mean = (velocity_with_localcsfe[:, 27, :]).reshape(-1)
    df['gx2+gy2_' + str(cutoff) + '_mean'] = local_csfe_9_mean

    return df

def generate_edge_datapoints(flag='p1'):
    # global width
    edge_box = [1207.4066, 308.07604, 609.89845]

    data_dir = 'data/sum-edge-p1-p2_10112023/'
    if flag=='p1':
        csfe_path = data_dir + 'csfe/sfe_local_nicocr_withxz_edge_p1.mat'
        velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p1.mat'
        nye13_path = data_dir + 'core_width_from_NYE/nye13_width_edge_p1.mat'
        nye33_path = data_dir + 'core_width_from_NYE/nye33_width_edge_p1.mat'
    elif flag=='p2':
        csfe_path = data_dir + 'csfe/delta_local_nicocr_withxz_edge_p2.mat'
        velocity_path = data_dir + 'velocity/nicocr_random_local_segment_velocity_withxz_edge_p2.mat'
        nye13_path = data_dir + 'core_width_from_NYE/nye13_width_edge_p2.mat'
        nye33_path = data_dir + 'core_width_from_NYE/nye33_width_edge_p2.mat'

    tempkey = [item for item in io.loadmat(nye13_path).keys() if '__' not in item]
    nye13 = io.loadmat(nye13_path)[tempkey[0]]
    nye13 = nye13[1:,:]
    tempkey = [item for item in io.loadmat(nye33_path).keys() if '__' not in item]
    nye33 = io.loadmat(nye33_path)[tempkey[0]]
    nye33 = nye33[1:,:]
    tempkey = [item for item in io.loadmat(csfe_path).keys() if '__' not in item]
    csfe = io.loadmat(csfe_path)[tempkey[0]]
    gradient_csfe_x = calculate_gradient(csfe,axis=0)
    plot_gradient_csfe_x = copy.deepcopy(gradient_csfe_x)
    gradient_csfe_y = calculate_gradient(csfe,axis=1)
    plot_gradient_csfe_y = copy.deepcopy(gradient_csfe_y)
    velocity = io.loadmat(velocity_path)['v00xy']

    vel_dis = velocity[:,2,:]

    csfe_df = pd.DataFrame()
    csfe_df['value']=csfe[:,2]
    print(csfe_df.describe())

    # create the periodic top and bottom for csfe, strain and stress
    top_csfe = copy.deepcopy(csfe)
    height = csfe[-1,1]-(2.*csfe[0,1]-csfe[1,1])
    top_csfe[:,1]+=height
    bottom_csfe = copy.deepcopy(csfe)
    bottom_csfe[:,1]-=height
    csfe1=np.concatenate([top_csfe,
                          csfe,
                          bottom_csfe])
    csfe = csfe1

    top_gradient_csfe_x = copy.deepcopy(gradient_csfe_x)
    top_gradient_csfe_x[:,1]+=height
    bottom_gradient_csfe_x = copy.deepcopy(gradient_csfe_x)
    bottom_gradient_csfe_x[:,1]-=height
    gradient_csfe_x1=np.concatenate([top_gradient_csfe_x,
                          gradient_csfe_x,
                          bottom_gradient_csfe_x])
    gradient_csfe_x = gradient_csfe_x1

    top_gradient_csfe_y = copy.deepcopy(gradient_csfe_y)
    top_gradient_csfe_y[:,1]+=height
    bottom_gradient_csfe_y = copy.deepcopy(gradient_csfe_y)
    bottom_gradient_csfe_y[:,1]-=height
    gradient_csfe_y1=np.concatenate([top_gradient_csfe_y,
                          gradient_csfe_y,
                          bottom_gradient_csfe_y])
    gradient_csfe_y = gradient_csfe_y1


    print('csfe shape ', csfe.shape)
    print('velocity shape', velocity.shape)
    print('nye13 shape', nye13.shape)
    print('nye33 shape', nye33.shape)

    #basic data_check
    print('csfe_x infor: ', np.amax(csfe[:,0]),  np.amin(csfe[:,0]))
    print('csfe_y infor: ', np.amax(csfe[:,1]), np.amin(csfe[:,1]))
    print('csfe_value infor: ', np.amax(csfe[:,2]), np.amin(csfe[:,2]))

    print('nye13_value infor: ', np.nanmax(nye13), np.nanmin(nye13), np.sum(np.isnan(nye13)))
    print('nye33_value infor: ', np.nanmax(nye33), np.nanmin(nye33), np.sum(np.isnan(nye33)))

    print('velocity_x infor: ', np.amax(velocity[:,0,:]), np.amin(velocity[:,0,:]))
    print('velocity_y infor: ', np.amax(velocity[:,1,:]), np.amin(velocity[:,1,:]))
    print('velocity_value infor: ', np.amax(velocity[:,2,:]), np.amin(velocity[:,2,:]), np.mean(np.abs(velocity[:,2,:])), np.sum(np.isnan(velocity)))
    print('csfe_value infor: ', np.amax(csfe[:,2]), np.amin(csfe[:,2]), np.sum(np.isnan(csfe[:,2])))
    print('------------------')

    velocity_with_localcsfe = sfe_feature(nye13, velocity,csfe,best_match=[0,1])
    velocity_x = (velocity_with_localcsfe[:,0,:]).reshape(-1)
    velocity_y = (velocity_with_localcsfe[:,1,:]).reshape(-1)
    velocity_value = (velocity_with_localcsfe[:,2,:]).reshape(-1)
    local_csfe = (velocity_with_localcsfe[:,4,:]).reshape(-1)

    top1_v = (velocity_with_localcsfe[:,6,:]).reshape(-1)
    down1_v = (velocity_with_localcsfe[:,7,:]).reshape(-1)
    top2_v = (velocity_with_localcsfe[:,8,:]).reshape(-1)
    down2_v = (velocity_with_localcsfe[:,9,:]).reshape(-1)
    top3_v = (velocity_with_localcsfe[:,10,:]).reshape(-1)
    down3_v = (velocity_with_localcsfe[:,11,:]).reshape(-1)
    top4_v = (velocity_with_localcsfe[:,12,:]).reshape(-1)
    down4_v = (velocity_with_localcsfe[:,13,:]).reshape(-1)
    top5_v = (velocity_with_localcsfe[:,14,:]).reshape(-1)
    down5_v = (velocity_with_localcsfe[:,15,:]).reshape(-1)
    top6_v = (velocity_with_localcsfe[:,16,:]).reshape(-1)
    down6_v = (velocity_with_localcsfe[:,17,:]).reshape(-1)
    top7_v = (velocity_with_localcsfe[:,18,:]).reshape(-1)
    down7_v = (velocity_with_localcsfe[:,19,:]).reshape(-1)
    top8_v = (velocity_with_localcsfe[:,20,:]).reshape(-1)
    down8_v = (velocity_with_localcsfe[:,21,:]).reshape(-1)
    top9_v = (velocity_with_localcsfe[:,22,:]).reshape(-1)
    down9_v = (velocity_with_localcsfe[:,23,:]).reshape(-1)

    local_nye13_max = (velocity_with_localcsfe[:,3,:]).reshape(-1)
    local_nye13_min = (velocity_with_localcsfe[:,24,:]).reshape(-1)
    local_nye13_mean = (velocity_with_localcsfe[:,25,:]).reshape(-1)
    local_nye13 = (velocity_with_localcsfe[:,26,:]).reshape(-1)
    #
    velocity_with_localcsfe = sfe_feature(nye33, velocity,csfe,best_match=[0,1])
    local_nye33_max = (velocity_with_localcsfe[:,3,:]).reshape(-1)
    local_nye33_min = (velocity_with_localcsfe[:,24,:]).reshape(-1)
    local_nye33_mean = (velocity_with_localcsfe[:,25,:]).reshape(-1)
    local_nye33 = (velocity_with_localcsfe[:,26,:]).reshape(-1)

    df = pd.DataFrame()
    df['local_velocity'] = velocity_value

    # df['fe_15'] = local_csfe
    df['nye13_max'] = local_nye13_max
    df['nye13_min'] = local_nye13_min
    df['nye13_mean'] = local_nye13_mean
    # df['nye13_range'] = local_nye13_max-local_nye13_min
    df['nye13'] = local_nye13
    df['nye33_max'] = local_nye33_max
    df['nye33_min'] = local_nye33_min
    df['nye33_mean'] = local_nye33_mean
    # df['nye33_range'] = local_nye33_max-local_nye33_min
    df['nye33'] = local_nye33
    df['top1_v'] = top1_v
    df['top2_v'] = top2_v
    df['top3_v'] = top3_v
    df['top4_v'] = top4_v
    df['top5_v'] = top5_v
    df['top6_v'] = top6_v
    df['top7_v'] = top7_v
    df['top8_v'] = top8_v
    df['top9_v'] = top9_v
    df['down1_v'] = down1_v
    df['down2_v'] = down2_v
    df['down3_v'] = down3_v
    df['down4_v'] = down4_v
    df['down5_v'] = down5_v
    df['down6_v'] = down6_v
    df['down7_v'] = down7_v
    df['down8_v'] = down8_v
    df['down9_v'] = down9_v

    for num_id in range(15):
        if num_id==0:
            df = construct_feature(df, nye=nye13, velocity=velocity, csfe=csfe,
                                   gradient_csfe_x=gradient_csfe_x,
                                   gradient_csfe_y=gradient_csfe_y,
                                   cutoff=[0, 1])
        else:
            # df = construct_feature(df,cutoff=[17*(num_id-0.5),17*(num_id+0.5)])
            df = construct_feature(df, nye=nye13, velocity=velocity, csfe=csfe,
                                   gradient_csfe_x=gradient_csfe_x,
                                   gradient_csfe_y=gradient_csfe_y,
                                   cutoff=[0, 17 * (num_id)])

    print(df.shape)
    df = df.dropna()

    print(df.shape)
    file_path = 'data/edge_'+flag+'_final.csv'
    df.to_csv(file_path,index=False)

if __name__ == "__main__":
    generate_edge_datapoints('p1')
    generate_edge_datapoints('p2')