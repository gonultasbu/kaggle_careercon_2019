#%%
import os
import time
import numpy as np
import pandas as pd
from seaborn import countplot, lineplot, barplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
import math

def quaternion_to_euler(x, y, z, w):

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
    actual['total_angular_velocity'] = np.sqrt(np.square(actual['angular_velocity_X']) + 
                                        np.square(actual['angular_velocity_Y']) + 
                                        np.square(actual['angular_velocity_Z']))
    actual['total_linear_acceleration'] = np.sqrt(np.square(actual['linear_acceleration_X']) + 
                                            np.square(actual['linear_acceleration_Y']) + 
                                            np.square(actual['linear_acceleration_Z']))
    
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


X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
SS = pd.read_csv('sample_submission.csv')

X_train.head()


#%%
le = LabelEncoder()
y_train['surface'] = le.fit_transform(y_train['surface'])

X_train = fe(X_train)
X_test = fe(X_test)

# Imputation   
X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)

X_train.replace(-np.inf, 0, inplace = True)
X_train.replace(np.inf, 0, inplace = True)
X_test.replace(-np.inf, 0, inplace = True)
X_test.replace(np.inf, 0, inplace = True)
X_train.head()

'''
#%%
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest Accuracy: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(
        criterion='gini',
        verbose=1,
        max_features='auto',
        max_leaf_nodes=None,
        random_state = 1337
        
    ),
    search_spaces = { 
        'max_depth': (1, 200),
        'n_estimators': (20, 300),
        'min_samples_split': (2,20),
        'min_samples_leaf': (1,10),
        'min_weight_fraction_leaf': (0.0,0.5),
        'min_impurity_decrease': (0,1E-7),
        
        
    },    
    cv = GroupShuffleSplit(n_splits=5, random_state = 1337),
    n_jobs = -1,
    n_iter = 40,   
    verbose = 1,
    refit = True,
    
)

# Fit the model
result = bayes_cv_tuner.fit(X_train, y_train['surface'], groups=y_train['group_id'], callback=status_print)

'''
#%%
folds = GroupShuffleSplit(n_splits=5, random_state=1337)
sub_preds = np.zeros((X_test.shape[0], 9))
score_sum = 0

clf = LGBMClassifier(colsample_bytree=0.03581900508076567, learning_rate=0.09308743154192588, 
                     max_bin=471, max_depth=-1, min_child_samples=50, min_child_weight=10, 
                     n_estimators=51, num_leaves=82, reg_alpha=2.0036361362515476e-07, reg_lambda=0.2069846982564452,
                     scale_pos_weight=9.05248363351552, subsample=0.9056112605172989, subsample_for_bin=368683, 
                     subsample_freq=9)
'''

'''
# clf = XGBClassifier() 
clf = RandomForestClassifier(max_depth=77, min_impurity_decrease=8.435373293206687e-08, 
                             min_samples_leaf=10, min_samples_split=4, min_weight_fraction_leaf=0.0, n_estimators=170, random_state=1337)

for i, (train_index, test_index) in enumerate(folds.split(X_train, y_train['surface'],
                                                groups=y_train['group_id'])):

    print('_'*20, i, '_'*20)
    clf.fit(X_train.iloc[train_index], y_train['surface'][train_index])
    score_sum += clf.score(X_train.iloc[test_index], y_train['surface'][test_index])
    print('train_score ', clf.score(X_train.iloc[train_index], y_train['surface'][train_index]))
    print('cv_score ', clf.score(X_train.iloc[test_index], y_train['surface'][test_index]))
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    features = X_train.columns
    
    '''
    hm = 30
    plt.figure(figsize=(7, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')
    plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    '''
    

print('_'*40)
mean_cv_acc = score_sum / folds.n_splits
print('Avg CV Accuracy', mean_cv_acc)
clf.fit(X_train, y_train['surface'])
sub_preds = clf.predict_proba(X_test)
SS['surface'] = le.inverse_transform(sub_preds.argmax(axis=1))
SS.to_csv('rf.csv', index=False)


#%%
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(clf.predict(X_train), y_train['surface'])
sns.heatmap(cm, annot=True, cmap="YlGnBu")


#%%
countplot(y = 'surface', data = y_train)
plt.show()


#%%
print(y_train['surface'].value_counts())


#%%
print(le.inverse_transform(y_train['surface'].unique()))
print(y_train['surface'].unique())


#%%
import time

print(int(time.time()))


#%%



