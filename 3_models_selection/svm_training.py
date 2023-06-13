import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df_dev = pd.read_csv('../dataset/development.csv')
df_eval = pd.read_csv('../dataset/evaluation.csv')

df = pd.concat([df_dev, df_eval], sort=False)

def final_preprocessing(df, reduce_df=True):
    df_preproc = df.copy()

    # one hot encoding
    enc = OneHotEncoder()
    encoded_df = pd.concat([df_preproc['weekday'], df_preproc['data_channel']], axis=1)
    enc.fit(encoded_df)
    encoded_df = enc.transform(encoded_df)
    additional_columns = enc.get_feature_names_out()
    df_preproc[additional_columns] = encoded_df.toarray()
    df_preproc.drop(['weekday', 'data_channel', 'url', 'id'], axis = 1, inplace=True)

    # drop from feature selection
    df_preproc.drop(columns=['n_non_stop_words', 'kw_min_min', 'kw_max_max'], inplace=True)

    if reduce_df:
        # remove n_tokens_content less than 0
        df_preproc = df_preproc.query("n_tokens_content > 0")
        df_preproc['n_tokens_content'] = np.log(df_preproc['n_tokens_content'])
        # Remove outliers from kw_avg_avg (we lost another 9% of the dataset)
        q1 = df_preproc['kw_avg_avg'].describe()['25%']
        q3 = df_preproc['kw_avg_avg'].describe()['75%']
        iqr = q3 - q1
        min_kw_avg_avg = q1 - 1.5*iqr
        max_kw_avg_avg = q3 + 1.5*iqr
        df_preproc = df_preproc[(df_preproc.kw_avg_avg < max_kw_avg_avg) & (df_preproc.kw_avg_avg > min_kw_avg_avg)]
    else:
        df_preproc['n_tokens_content'] = np.log(1 + df_preproc['n_tokens_content'])

    # adjust num_imgs, num_self_hrefs, num_videos, num_hrefs
    df_preproc['num_imgs'].fillna(df_preproc['num_imgs'].mean(), inplace=True)
    df_preproc['num_imgs'] = np.log(1 + df_preproc['num_imgs'])
    df_preproc['num_self_hrefs'].fillna(df_preproc['num_self_hrefs'].mean(), inplace=True)
    df_preproc['num_self_hrefs'] = np.log(1 + df_preproc['num_self_hrefs'])
    df_preproc['num_videos'].fillna(df_preproc['num_videos'].mean(), inplace=True)
    df_preproc['num_videos'] = np.log(1 + df_preproc['num_videos'])
    df_preproc['num_hrefs'] = np.log(1 + df_preproc['num_hrefs'])

    std_scaler = StandardScaler().fit(df_preproc[['n_tokens_title', 'n_tokens_content']])
    scaled_features = std_scaler.transform(df_preproc[['n_tokens_title', 'n_tokens_content']])
    df_preproc[['n_tokens_title', 'n_tokens_content']] = scaled_features

    df_preproc['avg_negative_polarity'] = df_preproc['avg_negative_polarity'].abs()

    # Since this features has a range between [0, 10], we can apply a min max scaling
    df_preproc['num_keywords'] = df.groupby(['data_channel'], sort=False)['num_keywords'].apply(lambda x: x.fillna(x.mean())).reset_index()['num_keywords']
    std_scaler = MinMaxScaler().fit(df_preproc[['num_keywords']])
    scaled_features = std_scaler.transform(df_preproc[['num_keywords']])
    df_preproc[['num_keywords']] = scaled_features

    if 'shares' in df_preproc:
        df_preproc['shares'] = np.log(df_preproc['shares'])

    std_scaler = StandardScaler().fit(df_preproc[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg', 'kw_max_avg', 'kw_max_min', 'kw_min_max']])
    scaled_features = std_scaler.transform(df_preproc[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']])
    df_preproc[['kw_avg_max', 'kw_avg_avg', 'kw_avg_min', 'kw_min_avg','kw_max_avg', 'kw_max_min', 'kw_min_max']] = scaled_features

    std_scaler = StandardScaler().fit(df_preproc[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']])
    scaled_features = std_scaler.transform(df_preproc[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']])
    df_preproc[['self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess']] = scaled_features


    is_weekend = []
    for _, row in df_preproc.iterrows():
        if row['weekday_sunday'] == 1 or row['weekday_saturday'] == 1:
            is_weekend.append(1)
        else:
            is_weekend.append(0)
    df_preproc['is_weekend'] = is_weekend

    std_scaler = StandardScaler().fit(df_preproc[['timedelta']])
    scaled_features = std_scaler.transform(df_preproc[['timedelta']])
    df_preproc[['timedelta']] = scaled_features

    return df_preproc

working_df_dev = final_preprocessing(df_dev)
working_df_eval = final_preprocessing(df_eval, reduce_df=False)

X = working_df_dev.drop(columns=["shares"]).values
y = working_df_dev["shares"].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=42)

params = {
    "kernel": ['rbf'],
    "gamma": ['scale', 'auto'],
    "tol": [1e-3, 1e-4],
    "C": [0.5, 1, 5, 10, 100],
    "epsilon": [0.3, 0.5, 1, 5, 10],
    "shrinking": [True, False]
}

gs = GridSearchCV(SVR(), param_grid=params, cv=5, scoring='r2', verbose=3, n_jobs=-1)
gs.fit(X_train, y_train)
print('==============================')
print(f'best_score_: {gs.best_score_}')
print('==============================')
print(f'best_params_: {gs.best_params_}')

svr = SVR(gs.best_params_)
svr.fit(X_valid, y_valid)

rms = mean_squared_error(y_valid, svr.predict(X_valid), squared=False)
print(f'root mean squared error: {rms}')

r2 = r2_score(y_valid, svr.predict(X_valid))
adj_r2 = 1-(1-r2)*(len(X_valid) - 1)/(len(X_valid) - X_valid.shape[1] - 1)
print(f'adjusted r2 score: {adj_r2}')

# Make final predictions
y_pred = gs.predict(working_df_eval.values)
final_preds = np.exp(y_pred)
# Write CSV
id_col = df_eval['id']
new_df = pd.DataFrame(columns=['Id', 'Predicted'])
new_df['Id'] = id_col
new_df['Predicted'] = final_preds
new_df.to_csv('../output/svm_results.csv', columns=['Id','Predicted'], index=False)