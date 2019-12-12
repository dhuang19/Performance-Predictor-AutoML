import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# CROSS VALIDATION ON HYPERPARAMETERS
#Create the parameter grid based on the results of random search 
param_grid_val = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100],
    'max_features': [2,3],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [2, 3],
    'n_estimators': [1400, 1600, 1800, 2000]
}

param_grid_train = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100],
    'max_features': [2,3],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [2, 3],
    'n_estimators': [1600, 1800, 2000, 2100]
}

rf_val = RandomForestRegressor()
# Instantiate the grid search model
grid_search_val = GridSearchCV(estimator = rf_val, param_grid = param_grid_val, 
                          cv = 2, n_jobs = 1, verbose = 2)

# Fit the grid search to the data
grid_search_val.fit(X, y[:, 0])
print(grid_search_val.best_params_)
#result: {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1400, 'min_samples_split': 3, 'max_features': 3, 'max_depth': 80}

rf_train = RandomForestRegressor()
# Instantiate the grid search model
grid_search_train = GridSearchCV(estimator = rf_train, param_grid = param_grid_train, 
                          cv = 2, n_jobs = 1, verbose = 2)

# Fit the grid search to the data
grid_search_train.fit(X, y[:, 1])
print(grid_search_train.best_params_)
#result: {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1600, 'min_samples_split': 2, 'max_features': 3, 'max_depth': 90}


#Runnin results of cross validation with random forest
model_val_error = RandomForestRegressor(bootstrap= True, min_samples_leaf=1, n_estimators=1400, max_features=3, min_samples_split=3, max_depth=80)
model_val_error.fit(X, y[:, 0])
pred_val_error = model_val_error.predict(test_data)

model_train_error = RandomForestRegressor(bootstrap=True, min_samples_leaf=1, n_estimators=1600, max_features=3, min_samples_split=2, max_depth=90)
model_train_error.fit(X, y[:, 1])
pred_train_error = model_train_error.predict(test_data)

pred = np.stack((pred_val_error, pred_train_error), axis=1)

print("R2 Score with cross validation: " + str(r2_score(test_sol, pred)))
print("Explained variance score with cross validation: " + str(explained_variance_score(test_sol, pred)))