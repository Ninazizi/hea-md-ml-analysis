import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

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
def cal_pearson(csfe_path,flag):
    df = pd.read_csv(csfe_path)
    print(df.shape)
    print(df.columns)
    target_col = 'ave_velocity'
    # target_col = 'local_velocity'
    if flag=='edge':
        df['ave_velocity']=(df['local_velocity']+df['top1_v']+df['top2_v']+df['down1_v']+df['down2_v'])/5.0
    elif flag=='screw':
        df['ave_velocity'] = (df['local_velocity'] + df['top1_v'] + df['top2_v'] +df['top3_v']+ \
                             df['down1_v'] + df['down2_v'] + df['down3_v'])/7.0
    feature_cols = [item for item in df.columns if 'fe' in item and 'mean' in item or 'g' in item] + [item for item in df.columns if 'nye' in item]
    # feature_cols = [item for item in df.columns if 'nye' in item]
    # feature_cols = [item for item in df.columns if 'fe' in item and 'mean' in item or 'gx2+gy2' in item]# + ['nye13','nye33']
    # feature_cols = [item for item in df.columns if 'gx2+gy2' in item ] #+ ['nye13','nye33']
    print(feature_cols)
    # Define the number of folds for cross-validation
    n_splits = 5
    # Convert the continuous target variable into discrete intervals for stratification
    df['target_col_binned'] = pd.cut(df[target_col], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store the results for each fold
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    rmse_base_scores = []
    mae_base_scores = []
    y_list = []
    true_list = []

    params = {
        'objective': 'regression',
        'metric': 'mse',
        # 'objective': 'binary',
        # 'metric': 'auc',
        # 'num_leaves': 30,
        'learning_rate': 0.05,
        'feature_fraction': 1,
        # 'max_depth':2
        # 'min_child_sample': 30,
    }

    for train_index, val_index in skf.split(df, df['target_col_binned']):
        # Split the data into training and validation sets
        train_data, val_data = df.iloc[train_index], df.iloc[val_index]

        # Filter the data for the features and target
        train_data = train_data[feature_cols + [target_col]]
        val_data = val_data[feature_cols + [target_col]]

        # Create LightGBM datasets
        train_dataset = lgb.Dataset(train_data[feature_cols].values, label=train_data[target_col].values)
        val_dataset = lgb.Dataset(val_data[feature_cols].values, label=val_data[target_col].values)

        # Train the LightGBM model
        model = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=10000,
            valid_sets=[train_dataset, val_dataset],
            early_stopping_rounds=100,
            verbose_eval=1000
        )

        # Predict on validation set
        y_val_hat = model.predict(val_data[feature_cols])

        # Calculate and store the scores
        mse_scores.append(mean_squared_error(val_data[target_col], y_val_hat, squared=True))
        rmse_scores.append(mean_squared_error(val_data[target_col], y_val_hat, squared=False))
        mae_scores.append(mean_absolute_error(val_data[target_col], y_val_hat))
        r2_scores.append(r2_score(val_data[target_col], y_val_hat))
        rmse_base_scores.append(
            mean_squared_error(val_data[target_col],
                               np.ones_like(val_data[target_col]) * np.mean(train_data[target_col]),
                               squared=False))
        mae_base_scores.append(
            mean_absolute_error(val_data[target_col],
                                np.ones_like(val_data[target_col]) * np.mean(train_data[target_col])))
        y_list.append(y_val_hat)
        true_list.append(val_data[target_col])

    # Print the average scores across all folds
    print("Average MSE: ", sum(mse_scores) / n_splits)
    print("Average RMSE: ", sum(rmse_scores) / n_splits)
    print("Average MAE: ", sum(mae_scores) / n_splits)
    print("Average r2: ", sum(r2_scores) / n_splits)
    print("Average RMSE_base: ", sum(rmse_base_scores) / n_splits)
    print("Average MAE_base: ", sum(mae_base_scores) / n_splits)
    # Drop the binned column after cross-validation
    df.drop('target_col_binned', axis=1, inplace=True)

    return (np.concatenate(y_list), np.concatenate(true_list), r2_scores)


flag = 'edge'
if flag=='edge':
    data_dir = '1011/sum-edge-p1-p2_10112023/'
    # height = edge_box[2]
else:
    data_dir = '1011/sum-screw-p1-p2_10112023/'
    # height = screw_box[2]

data_dir = 'data/'
csfe_path = data_dir + 'edge_p1_final.csv'
y_edge1 = cal_pearson(csfe_path,flag='edge')

csfe_path = data_dir + 'edge_p2_final.csv'
y_edge2 = cal_pearson(csfe_path,flag='edge')

csfe_path = data_dir + 'screw_p1_final.csv'
y_screw1 = cal_pearson(csfe_path,flag='screw')

csfe_path = data_dir + 'screw_p2_final.csv'
y_screw2 = cal_pearson(csfe_path,flag='screw')
print('edge_leading r2,',np.mean(y_edge1[2]))
print('edge_trailing r2,',np.mean(y_edge2[2]))
print('screw_leading r2,',np.mean(y_screw1[2]))
print('screw_leading r2,',np.mean(y_screw2[2]))

# Assuming 'split_r2_scores' is a list of lists containing R2 scores for each split
split_r2_scores = [y_edge1[2], y_edge2[2], y_screw1[2], y_screw2[2]]
labels = ['leading_edge', 'trailing_edge', 'leading_screw', 'trailing_screw']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

num_splits = len(split_r2_scores[0])
num_categories = len(split_r2_scores)
bar_width = 0.1  # Width of the bars
index = np.arange(num_categories)

# Plotting the histograms
for i in range(num_splits):
    plt.bar(index + i * bar_width, [scores[i] for scores in split_r2_scores],
            bar_width, alpha=0.5, label=f'Split {i+1}')

# plt.xlabel('Category')
plt.ylabel('Prediction accuracy',weight='bold',size=13)
plt.ylim([0,1])
# plt.title('R2 Scores by Category for Each Split')
plt.xticks(index + bar_width * (num_splits / 2), labels,weight='bold',size=13)
plt.legend()
plt.tight_layout()
plt.savefig('fig13-2.jpg')

x_min = np.amin(np.concatenate([y_edge1[0],y_edge1[1],y_edge2[0],y_edge2[1],y_screw1[0],y_screw1[1],y_screw2[0],y_screw2[1]]))
x_max = np.amax(np.concatenate([y_edge1[0],y_edge1[1],y_edge2[0],y_edge2[1],y_screw1[0],y_screw1[1],y_screw2[0],y_screw2[1]]))
fig,axs = plt.subplots(2,2,figsize=(7, 7),sharex=True,sharey=True)
# Calculate ticks based on x_min and x_max
ticks = np.arange(-2, 14, (14 + 2) / 8)

# Set the same ticks for both x and y axis
for ax in axs.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

axs[0,0].set_title('leading_edge',weight='bold',size=13)
axs[0,0].scatter(y_edge1[0],y_edge1[1], alpha=0.2,s=6)
axs[0,0].plot(np.arange(x_min,x_max,0.1),np.arange(x_min,x_max,0.1),c='gray',linewidth=1,linestyle='dashed',)

axs[0,1].set_title('trailing_edge',weight='bold',size=13)
axs[0,1].scatter(y_edge2[0],y_edge2[1], alpha=0.2,s=6)
axs[0,1].plot(np.arange(x_min,x_max,0.1),np.arange(x_min,x_max,0.1),c='gray',linewidth=1,linestyle='dashed',)

axs[1,0].set_title('leading_screw',weight='bold',size=13)
axs[1,0].scatter(y_screw1[0],y_screw1[1], alpha=0.2,s=6)
axs[1,0].plot(np.arange(x_min,x_max,0.1),np.arange(x_min,x_max,0.1),c='gray',linewidth=1,linestyle='dashed',)

axs[1,1].set_title('trailing_screw',weight='bold',size=13)
axs[1,1].scatter(y_screw2[0],y_screw2[1], alpha=0.2,s=6)
axs[1,1].plot(np.arange(x_min,x_max,0.1),np.arange(x_min,x_max,0.1),c='gray',linewidth=1,linestyle='dashed',)

plt.tight_layout()
plt.savefig('fig14.jpg')

