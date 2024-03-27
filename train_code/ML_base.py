import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import os
import random

# ML
from sklearn.ensemble import RandomForestRegressor  # Bagging
from xgboost.sklearn import XGBRFRegressor           # GBM

# KFold(CV), partial : for optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from functools import partial
# from imblearn.over_sampling import SMOTE

# AutoML framework
import optuna
from optuna.samplers import TPESampler

from sklearn.metrics import mean_absolute_percentage_error

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def csv_preprocessing(df):
    lb = LabelEncoder()
    df.sex = lb.fit_transform(df.sex)  # M->0, F->1
    train = df.drop(columns=["ID","filename","ext_tooth","date"])

    # 결측치 채우기
    df.loc[df['height'] != df['height'], 'height'] = df['height'].mean()
    df.loc[df['weight'] != df['weight'], 'weight'] = df['weight'].mean()
    df.loc[df['bmi'] != df['bmi'], 'bmi'] = df['bmi'].mean()

    df[['A','K','P','Y']] = pd.get_dummies(df['operator'], dtype=int)
    df = df.drop(columns=["operator"])
    
    return df

# randomforest
def rf_optimizer(trial, X, y, K):
    # define parameter to tune
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200])
    max_depth = trial.suggest_int('max_depth', 4, 10)
    max_features = trial.suggest_categorical('max_features', [0.6, 0.7, 0.8])
    
    
    # set model
    model = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   max_features=max_features,
                                   criterion='absolute_error', # log_loss
                                #    class_weight='balanced'
                                  )
    
    # K-Fold Cross validation
    folds = StratifiedKFold(n_splits=K, shuffle=True)
    losses = []
    
    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = mean_absolute_percentage_error(y_val, preds)
        losses.append(loss)
    
    
    # return mean score of CV
    return np.mean(losses)

# xgboost
def xgb_optimizer(trial, X, y, K):
    n_estimators = trial.suggest_categorical('n_estimators', [500, 1000, 2000])
    max_depth = trial.suggest_int('max_depth', 4, 10)
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8])
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2)
    reg_lambda = trial.suggest_categorical('reg_lambda', [0.1, 0.5, 1, 2])
    
    
    model = XGBRFRegressor(n_estimators=n_estimators,
                          max_depth=max_depth,
                          colsample_bytree=colsample_bytree,
                          learning_rate=learning_rate,
                          reg_lambda=reg_lambda)
    
    
    folds = StratifiedKFold(n_splits=K, shuffle=True)
    losses = []
    
    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = mean_absolute_percentage_error(y_val, preds)
        losses.append(loss)
    
    
    return np.mean(losses)

# set configs
seed = 42
seed_everything(seed)

is_tuning = True
if is_tuning:
    sampler = TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.
is_scaling = True
is_pca = False
if is_tuning:
    n_trials=30
K = 3 # set K of K-Fold

####################################  data  ###################################
train = pd.read_csv('/DATA/train/train.csv')
test = pd.read_csv('/DATA/test/test.csv')
submission = pd.read_csv('/DATA/test/sample_submission.csv')

train = csv_preprocessing(train)
test = csv_preprocessing(test)

# feature selection
X = train.drop(columns=["time_min", "filename", "date", "ID"])
y = train['time_min']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=seed)

if is_scaling:
    scaler = StandardScaler()
    data_ = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(data=data_, columns=X_train.columns)
    data_ = scaler.transform(X_val)
    X_val = pd.DataFrame(data=data_, columns=X_val.columns)

if is_pca:
    pca = PCA(n_components=0.90, random_state=seed)
    data_ = pca.fit_transform(X_train)
    X_train = pd.DataFrame(data=data_, columns=[f"PC{i}" for i in range(1, data_.shape[1]+1)])
    data_ = pca.transform(X_val)
    X_val = pd.DataFrame(data=data_, columns=[f"PC{i}" for i in range(1, data_.shape[1]+1)])

opt_func = partial(rf_optimizer, X=X, y=y, K=K)

if is_tuning:
    rf_study = optuna.create_study(direction="minimize", sampler=sampler) # determine minimize or maximize sth
    rf_study.optimize(opt_func, n_trials=n_trials)
    
opt_func = partial(xgb_optimizer, X=X, y=y, K=K)

if is_tuning:
    xgb_study = optuna.create_study(direction="minimize", sampler=sampler)
    xgb_study.optimize(opt_func, n_trials=n_trials)

# Finalize Models
if is_tuning:
    rf_best_params = rf_study.best_params
    xgb_best_params = xgb_study.best_params

    best_rf = RandomForestRegressor(**rf_best_params)
    best_xgb = XGBRFRegressor(**xgb_best_params)

best_rf.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)

# OOF-prediction
v_rf = best_rf.predict(X_val)
v_xgb = best_xgb.predict(X_val)

voting_weights = [0.7, 0.3]

ensembles = voting_weights[0]*v_rf[:] + voting_weights[1]*v_xgb[:]

print("rf Best Score: %.4f" % rf_study.best_value)
print("xgb Best Score: %.4f" % xgb_study.best_value)
print("(After finalization)OOF prediction MAPE : %.4f" % mean_absolute_percentage_error(y_val, ensembles))

####################################  test  ###################################
# test data preprocessing in same way
X_test = test[X.columns].fillna(X.mean())
if is_scaling:
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(data=X_test, columns=X.columns)

if is_pca:
    data_ = pca.transform(X_test)
    X_test = pd.DataFrame(data=data_, columns=[f"PC{i}" for i in range(1, data_.shape[1]+1)])
# predict test data
preds_rf = best_rf.predict(X_test)
preds_xgb = best_xgb.predict(X_test)

# make submission.csv
submission['time_min'] = voting_weights[0]*preds_rf[:] + voting_weights[1]*preds_xgb[:]
submission.to_csv("/USER/baseline/data/submission.csv", index=False)