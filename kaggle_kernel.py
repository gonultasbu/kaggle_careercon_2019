#%% [markdown]
# # Robots are smart… by design !!
# 
# Who make those Robots smart? Its you Machine Learning guys !
# In this project, our task is to help robots recognize the floor surface they’re standing on using data collected from Inertial Measurement Units (IMU sensors).
# 
# Hope you guys will learn something from this sensor data. Its kind of IOT data, as in IOT, we usually work with sensor data..  
# 
# ## Its a golden chance to help humanity, by helping Robots !

#%%
import os
import time
import numpy as np
import pandas as pd
from seaborn import countplot,lineplot, barplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


#%%
tr = pd.read_csv('X_train.csv')
te = pd.read_csv('X_test.csv')
target = pd.read_csv('y_train.csv')
ss = pd.read_csv('sample_submission.csv')


#%%
tr.head()


#%%
tr.shape, te.shape


#%%
countplot(y = 'surface', data = target)
plt.show()

#%% [markdown]
# We need to classify on which surface our robot is standing.
# 
# So, its a simple classification task. Multi-class to be specific.

#%%
len(tr.measurement_number.value_counts())

#%% [markdown]
# What's that?
# Each series has 128 measurements. 

#%%
tr.shape[0] / 128, te.shape[0] / 128

#%% [markdown]
# So, we have 3810 train series, and 3816 test series.
# Let's engineer some features!
#%% [markdown]
# ## Feature Engineering

#%%
# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def fe(actual):
    new = pd.DataFrame()
    actual['total_angular_velocity'] = actual['angular_velocity_X'] + actual['angular_velocity_Y'] + actual['angular_velocity_Z']
    actual['total_linear_acceleration'] = actual['linear_acceleration_X'] + actual['linear_acceleration_Y'] + actual['linear_acceleration_Z']
    
    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']
    
    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    
    def f1(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    def f2(x):
        return np.mean(np.abs(np.diff(x)))
    
    for col in actual.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()
        new[col + '_min'] = actual.groupby(['series_id'])[col].min()
        new[col + '_max'] = actual.groupby(['series_id'])[col].max()
        new[col + '_std'] = actual.groupby(['series_id'])[col].std()
        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']
        
        # Change. 1st order.
        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)
        
        # Change of Change. 2nd order.
        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)
        
        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))

    return new


#%%
get_ipython().run_cell_magic('time', '', 'tr = fe(tr)\nte = fe(te)\ntr.head()')


#%%
tr.head()


#%%
le = LabelEncoder()
target['surface'] = le.fit_transform(target['surface'])


#%%
tr.fillna(0, inplace = True)
te.fillna(0, inplace = True)


#%%
tr.replace(-np.inf, 0, inplace = True)
tr.replace(np.inf, 0, inplace = True)
te.replace(-np.inf, 0, inplace = True)
te.replace(np.inf, 0, inplace = True)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
sub_preds = np.zeros((te.shape[0], 9))
score = 0
for i, (train_index, test_index) in enumerate(folds.split(tr, target['surface'])):
    print('-'*20, i, '-'*20)
    
    clf =  lgb.LGBMClassifier(device='gpu')
    clf.fit(tr.iloc[train_index], target['surface'][train_index])
    
    sub_preds += clf.predict_proba(te) / folds.n_splits
    score += clf.score(tr.iloc[test_index], target['surface'][test_index])
    print('score ', clf.score(tr.iloc[test_index], target['surface'][test_index]))
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    features = tr.columns

    hm = 30
    plt.figure(figsize=(7, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')
    plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

print('Avg Accuracy', score / folds.n_splits)

#%%
ss['surface'] = le.inverse_transform(sub_preds.argmax(axis=1))
ss.to_csv('rf.csv', index=False)
# ss.head(10)

#%% [markdown]
# ## To be Continued..

