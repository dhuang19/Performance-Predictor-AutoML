import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# RANDOM GRID RANDOM REGRESSOR
# Data set of training data that's been split into training & test
# X = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/split_train_data.csv')
# y = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/split_train_labels.csv')
# y = np.array(y)[:,1:]
# test_data = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/split_test_data')
# test_sol = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/split_test_labels.csv')

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# random regressor for val_error
# rf_val = RandomForestRegressor()
# rf_random_val = RandomizedSearchCV(estimator = rf_val, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# rf_random_val.fit(X, y[:, 0])

# print(rf_random_val.best_params_)
# result: {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 1800, 'max_features': 'auto', 'min_samples_split': 2, 'max_depth': None}

# random regressor for train_error
# rf_train = RandomForestRegressor()
# rf_random_train = RandomizedSearchCV(estimator = rf_train, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# rf_random_train.fit(X, y[:, 1])

# print(rf_random_train.best_params_)
# result: {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 2000, 'max_features': 'auto', 'min_samples_split': 2, 'max_depth': 90}

# Running random forest with results collected from random regressor
model_val_error = RandomForestRegressor(bootstrap= True, min_samples_leaf=2, n_estimators=1800, max_features='auto', min_samples_split=2, max_depth=None)
model_val_error.fit(X, y[:, 0])
pred_val_error = model_val_error.predict(test_data)

model_train_error = RandomForestRegressor(bootstrap=True, min_samples_leaf=2, n_estimators=2000, max_features='auto', min_samples_split=2, max_depth=90)
model_train_error.fit(X, y[:, 1])
pred_train_error = model_train_error.predict(test_data)

pred = np.stack((pred_val_error, pred_train_error), axis=1)

print("R2 Score w/o cross validation: " + str(r2_score(test_sol, pred)))
print("Explained variance score w/o cross validation: " + str(explained_variance_score(test_sol, pred)))

# CROSS VALIDATION ON HYPERPARAMETERS
# Create the parameter grid based on the results of random search 
# param_grid_val = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100],
#     'max_features': [2,3],
#     'min_samples_leaf': [1,2,3],
#     'min_samples_split': [2, 3],
#     'n_estimators': [1400, 1600, 1800, 2000]
# }

# param_grid_train = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100],
#     'max_features': [2,3],
#     'min_samples_leaf': [1,2,3],
#     'min_samples_split': [2, 3],
#     'n_estimators': [1600, 1800, 2000, 2100]
# }

# rf_val = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search_val = GridSearchCV(estimator = rf_val, param_grid = param_grid_val, 
#                           cv = 2, n_jobs = 1, verbose = 2)

# # Fit the grid search to the data
# grid_search_val.fit(X, y[:, 0])
# print(grid_search_val.best_params_)
# result: {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1400, 'min_samples_split': 3, 'max_features': 3, 'max_depth': 80}

# rf_train = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search_train = GridSearchCV(estimator = rf_train, param_grid = param_grid_train, 
#                           cv = 2, n_jobs = 1, verbose = 2)

# # Fit the grid search to the data
# grid_search_train.fit(X, y[:, 1])
# print(grid_search_train.best_params_)
# result: {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1600, 'min_samples_split': 2, 'max_features': 3, 'max_depth': 90}


# Runnin results of cross validation with random forest
model_val_error = RandomForestRegressor(bootstrap= True, min_samples_leaf=1, n_estimators=1400, max_features=3, min_samples_split=3, max_depth=80)
model_val_error.fit(X, y[:, 0])
pred_val_error = model_val_error.predict(test_data)

model_train_error = RandomForestRegressor(bootstrap=True, min_samples_leaf=1, n_estimators=1600, max_features=3, min_samples_split=2, max_depth=90)
model_train_error.fit(X, y[:, 1])
pred_train_error = model_train_error.predict(test_data)

pred = np.stack((pred_val_error, pred_train_error), axis=1)

print("R2 Score with cross validation: " + str(r2_score(test_sol, pred)))
print("Explained variance score with cross validation: " + str(explained_variance_score(test_sol, pred)))

# Start automl on actual test csv
# X = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_norm_stats.csv')
# y = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_labels_norm_stats.csv')
# y = np.array(y)[:,1:]

# test_data = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_testing_data_norm_stats.csv')

# Uing params from random regressor
# model_val_error = RandomForestRegressor(bootstrap= True, min_samples_leaf=2, n_estimators=1800, max_features='auto', min_samples_split=2, max_depth=None)
# model_val_error.fit(X, y[:, 0])
# pred_val_error = model_val_error.predict(test_data)

# Using params from random regressor
# model_train_error = RandomForestRegressor(bootstrap=True, min_samples_leaf=2, n_estimators=2000, max_features='auto', min_samples_split=2, max_depth=90)
# model_train_error.fit(X, y[:, 1])
# pred_train_error = model_train_error.predict(test_data)

# pred = np.stack((pred_val_error, pred_train_error), axis=1)

# # Put into csv file
# # 2nd col: test_{0 -> 475}_val_error | test_{0-> 475}_train_error
# col1 = np.empty(952, dtype=object)
# col2 = np.empty(952, dtype=float)

# pred_row_idx = 0
# for i in range(952):
#     #print("i : " +  str(i) + "  pred_row_idx : " + str(pred_row_idx))
#     # At odd idx...train_error
#     if ((i%2) != 0):
#         col1[i] = "test_"+ str(pred_row_idx) + "_train_error"
#         col2[i] = pred[pred_row_idx][1]
#         pred_row_idx = pred_row_idx+1
#     # At even idx...val_error
#     else:
#         col1[i] = "test_"+ str(pred_row_idx) + "_val_error"
#         col2[i] = pred[pred_row_idx][0]

# pred_dict = { 'id': col1,
#             'Predicted': col2 }
# pred_df = pd.DataFrame(pred_dict, columns=['id', 'Predicted'])
# #print(pred_df)
# pred_df.to_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/submission.csv', index=False)
