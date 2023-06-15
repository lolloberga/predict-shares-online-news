import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFECV

df_dev = pd.read_csv('../dataset/development.csv')
df_eval = pd.read_csv('../dataset/evaluation.csv')

df = pd.concat([df_dev, df_eval], sort=False)


def final_preprocessing_eval(df, dev_stats):
    working_df_dev = df.copy()

    for index, row in working_df_dev.iterrows():
        if 'data_channel' in row and not row['num_keywords'] >= 0:
            working_df_dev.at[index, 'num_keywords'] = dev_stats['num_keywords_mean'][row['data_channel']]
            
    enc = OneHotEncoder()
    encoded_df = pd.concat([df['weekday'], df['data_channel']], axis=1)
    enc.fit(encoded_df)
    encoded_df = enc.transform(encoded_df)
    additional_columns = enc.get_feature_names_out()
    working_df_dev[additional_columns] = encoded_df.toarray()
    working_df_dev.drop(['weekday', 'data_channel', 'url', 'id'], axis = 1, inplace=True)



    working_df_dev['n_tokens_content'] = np.log(1 + working_df_dev['n_tokens_content'])

    std_scaler = dev_stats['kw_scaler']
    scaled_features = std_scaler.transform(working_df_dev[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']])
    working_df_dev[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']] = scaled_features

    std_scaler = dev_stats['ref_scaler']
    scaled_features = std_scaler.transform(working_df_dev[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']])
    working_df_dev[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']] = scaled_features

    std_scaler = dev_stats['scaler_details']
    scaled_features = std_scaler.transform(working_df_dev[['n_tokens_title', 'n_tokens_content']])
    working_df_dev[['n_tokens_title', 'n_tokens_content']] = scaled_features

    working_df_dev['num_imgs'].fillna(dev_stats['num_imgs_mean'], inplace=True)
    working_df_dev['num_imgs'] = np.log(1 + working_df_dev['num_imgs'])

    working_df_dev['num_self_hrefs'].fillna(dev_stats['num_self_hrefs_mean'], inplace=True)
    working_df_dev['num_self_hrefs'] = np.log(1 + working_df_dev['num_self_hrefs'])

    working_df_dev['num_videos'].fillna(dev_stats['num_videos_mean'], inplace=True)
    working_df_dev['num_videos'] = np.log(1 + working_df_dev['num_videos'])

    is_weekend = []
    for _, row in working_df_dev.iterrows():
        if row['weekday_sunday'] == 1 or row['weekday_saturday'] == 1:
            is_weekend.append(1)
        else:
            is_weekend.append(0)
    working_df_dev['is_weekend'] = is_weekend
    working_df_dev.drop(columns=[x for x in additional_columns if x.startswith('weekday')], inplace=True)

    std_scaler = dev_stats['time_scaler']
    scaled_features = std_scaler.transform(working_df_dev[['timedelta']])
    working_df_dev[['timedelta']] = scaled_features

    # std_scaler = dev_stats['scaler']
    # scaled_features = std_scaler.transform(working_df_dev)
    # working_df_dev[:] = scaled_features[:]

    return working_df_dev

def final_preprocessing_dev(df):
    working_df_dev = df.copy()
    dev_stats = dict()

    enc = OneHotEncoder()
    encoded_df = pd.concat([df['weekday'], df['data_channel']], axis=1)
    enc.fit(encoded_df)
    encoded_df = enc.transform(encoded_df)
    additional_columns = enc.get_feature_names_out()
    working_df_dev[additional_columns] = encoded_df.toarray()
    working_df_dev.drop(['weekday', 'data_channel', 'url', 'id'], axis = 1, inplace=True)

    working_df_dev['num_keywords'] = df.groupby(['data_channel'], sort=False)['num_keywords'].apply(lambda x: x.fillna(x.mean())).reset_index()['num_keywords']
    dev_stats['num_keywords_mean'] = df.groupby(['data_channel'], sort=False)['num_keywords'].mean()

    working_df_dev['n_tokens_content'] = np.log(1 + working_df_dev['n_tokens_content'])

    working_df_dev['shares'] = np.log(working_df_dev['shares'])

    # Remove outliers from kw_avg_avg (we lost another 9% of the dataset)
    q1 = working_df_dev['kw_avg_avg'].describe()['25%']
    q3 = working_df_dev['kw_avg_avg'].describe()['75%']
    iqr = q3 - q1
    min_kw_avg_avg = q1 - 1.5*iqr
    max_kw_avg_avg = q3 + 1.5*iqr
    working_df_dev = working_df_dev[(df.kw_avg_avg < max_kw_avg_avg) & (df.kw_avg_avg > min_kw_avg_avg)]

    std_scaler = StandardScaler().fit(working_df_dev[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg', 'kw_max_avg', 'kw_max_min', 'kw_min_max']])
    scaled_features = std_scaler.transform(working_df_dev[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']])
    working_df_dev[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']] = scaled_features
    dev_stats['kw_scaler'] = std_scaler

    std_scaler = StandardScaler().fit(working_df_dev[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']])
    scaled_features = std_scaler.transform(working_df_dev[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']])
    working_df_dev[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']] = scaled_features
    dev_stats['ref_scaler'] = std_scaler

    std_scaler = StandardScaler().fit(working_df_dev[['n_tokens_title', 'n_tokens_content']])
    scaled_features = std_scaler.transform(working_df_dev[['n_tokens_title', 'n_tokens_content']])
    working_df_dev[['n_tokens_title', 'n_tokens_content']] = scaled_features
    dev_stats['scaler_details'] = std_scaler

    working_df_dev['num_imgs'].fillna(working_df_dev['num_imgs'].mean(), inplace=True)
    working_df_dev['num_imgs'] = np.log(1 + working_df_dev['num_imgs'])
    dev_stats['num_imgs_mean'] = working_df_dev['num_imgs'].mean()

    working_df_dev['num_self_hrefs'].fillna(working_df_dev['num_self_hrefs'].mean(), inplace=True)
    working_df_dev['num_self_hrefs'] = np.log(1 + working_df_dev['num_self_hrefs'])
    dev_stats['num_self_hrefs_mean'] = working_df_dev['num_self_hrefs'].mean()

    working_df_dev['num_videos'].fillna(working_df_dev['num_videos'].mean(), inplace=True)
    working_df_dev['num_videos'] = np.log(1 + working_df_dev['num_videos'])
    dev_stats['num_videos_mean'] = working_df_dev['num_videos'].mean()

    is_weekend = []
    for _, row in working_df_dev.iterrows():
        if row['weekday_sunday'] == 1 or row['weekday_saturday'] == 1:
            is_weekend.append(1)
        else:
            is_weekend.append(0)
    working_df_dev['is_weekend'] = is_weekend
    working_df_dev.drop(columns=[x for x in additional_columns if x.startswith('weekday')], inplace=True)

    std_scaler = StandardScaler().fit(working_df_dev[['timedelta']])
    scaled_features = std_scaler.transform(working_df_dev[['timedelta']])
    working_df_dev[['timedelta']] = scaled_features
    dev_stats['time_scaler'] = std_scaler
    # features_for_scale = working_df_dev.drop(columns=['shares'])
    # std_scaler = StandardScaler().fit(features_for_scale)
    # scaled_features = std_scaler.transform(features_for_scale)
    # dev_stats['scaler'] = std_scaler
    
    # features_for_scale[:] = scaled_features[:]
    # features_for_scale['shares'] = working_df_dev['shares']
    # std_scaler = StandardScaler().fit(working_df_dev[['shares']])
    # scaled_features = std_scaler.transform(working_df_dev[['shares']])
    # working_df_dev[['shares']] = scaled_features

    return working_df_dev, dev_stats

def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res['v1'].tolist() + res['v2'].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res['drop'].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
    q = list(set(p['v1'].tolist() + p['v2'].tolist()))
    drop = (list(set(q).difference(set(keep))))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop']))
    for item in more_drop:
        drop.append(item)

    return drop

def corrX_new(df, cut = 0.9):
    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis = 1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool_))
    dropcols = list()

    res = pd.DataFrame(columns=(['v1', 'v2', 'v1.target',
                                 'v2.target','corr', 'drop' ]))
    for row in range(len(up)-1):
        col_idx = row + 1
        for col in range (col_idx, len(up)):
            if corr_mtx.iloc[row, col] > cut:
                if avg_corr.iloc[row] > avg_corr.iloc[col]:
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else:
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]

                s = pd.Series([ corr_mtx.index[row],
                                up.columns[col],
                                avg_corr[row],
                                avg_corr[col],
                                up.iloc[row,col],
                                drop],
                              index = res.columns)

                res.loc[len(res)] = s.to_numpy()

    dropcols_names = calcDrop(res)

    return dropcols_names

working_df_dev, dev_stats = final_preprocessing_dev(df_dev)
df_working_df_eval = final_preprocessing_eval(df_eval, dev_stats)

drop_new = corrX_new(working_df_dev, cut = 0.65)
working_df_dev.drop(drop_new, axis=1, inplace=True)
df_working_df_eval.drop(drop_new, axis=1, inplace=True)

# best_params = {'eta': 0.025,  'eval_metric': 'rmse',  'max_depth': 8,  'min_child_weight': 25,  'n_estimators': 300}
best_params = {'learning_rate': 0.05, 'loss': 'squared_error', 'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 200, 'random_state': 42}


X = working_df_dev.drop(columns=["shares"]).values
y = working_df_dev["shares"].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=42)

df_feature_train = pd.DataFrame(X_train)
# y_train = pd.DataFrame(y_train)

trans = RFECV(estimator=RandomForestRegressor(), step=1, cv=4 ,n_jobs=-1, verbose=2, scoring='neg_root_mean_squared_error')
trans.fit(df_feature_train, y_train)
X_trans_train = trans.transform(df_feature_train)
# print(f'trans.feature_names_in_: {trans.feature_names_in_}')
# print(len(trans.feature_names_in_))
# these are the indices of the columns that are considered important
print(f'initial shape: {df_feature_train.shape}')
print(f'new number of features: {trans.n_features_}')

# xgbr = XGBRegressor(**best_params)
grad_b = GradientBoostingRegressor(**best_params)
grad_b.fit(X_train, y_train)

df_feature_valid = pd.DataFrame(X_valid)
y_valid = pd.DataFrame(y_valid)
X_valid_trans = trans.transform(df_feature_valid)


rms = mean_squared_error(y_valid, grad_b.predict(X_valid), squared=False)
print(rms)
r2 = r2_score(y_valid, grad_b.predict(X_valid))
adj_r2 = 1-(1-r2)*(len(X_valid) - 1)/(len(X_valid) -X_valid.shape[1] - 1)
print(adj_r2)

df_feature = pd.DataFrame(X)


X_trans = trans.transform(df_feature)
eval_trans = trans.transform(df_working_df_eval)


# xgbr = XGBRegressor(**best_params)
grad_b = GradientBoostingRegressor(**best_params)
grad_b.fit(X, y)

# Make final predictions
y_pred = grad_b.predict(df_working_df_eval)
final_preds = np.exp(y_pred)
# Write CSV
id_col = df_eval['id']
new_df = pd.DataFrame(columns=['Id', 'Predicted'])
new_df['Id'] = id_col
new_df['Predicted'] = final_preds
print(new_df.describe())
new_df.to_csv('../output/grad_b_with_rfecv.csv', columns=['Id','Predicted'], index=False)