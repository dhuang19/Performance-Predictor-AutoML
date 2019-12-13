import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/train_OG.csv')
Y_train = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/labels_OG.csv')
X_test = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/test_OG.csv')

Y_train_valE = Y_train.iloc[:, 0]
Y_train_trainE = Y_train.iloc[:, 1]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random regressor for val_error
# rf_val = RandomForestRegressor()
# rf_random_val = RandomizedSearchCV(estimator = rf_val, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# rf_random_val.fit(X_train, Y_train_valE)
# # result: {'bootstrap': True, 'min_samples_leaf': 4, 'n_estimators': 270, 'max_features': 'auto', 'min_samples_split': 2, 'max_depth': 100

# print(rf_random_val.best_params_)

# rf_train = RandomForestRegressor()
# rf_random_train = RandomizedSearchCV(estimator = rf_train, param_distributions = random_grid, n_iter = 30, cv = 3, verbose=2, random_state=42)
# rf_random_train.fit(X_train, Y_train_trainE)

# print(rf_random_train.best_params_)
# result: {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 1000, 'max_features': 'auto', 'min_samples_split': 5, 'max_depth': 60}