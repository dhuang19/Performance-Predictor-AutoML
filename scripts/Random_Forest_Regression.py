import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Runnin random forest with results from random regressor on train_test_split data to predict train_error
X = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_norm_stats.csv')
y = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_labels_norm_stats.csv')
y = np.array(y)[:,1:]

state = np.random.randint(0, 2838654)
# X_train, X_test, y_train, y_test = \
#                          train_test_split(X, y[:,1], test_size=0.33, random_state=state)

# y_test, y_train = np.array(y_test), np.array(y_train)

# clf = RandomForestRegressor(bootstrap=True, min_samples_leaf=2, n_estimators=2000, max_features='auto', min_samples_split=2, max_depth=90)
# clf.fit(X_train, y_train)
# # print(np.argmax(clf.feature_importances_))
# pred = clf.predict(X_test)
# print(explained_variance_score(pred, y_test))
# print(r2_score(pred, y_test))

# Runnin random forest with results from random regressor on train_test_split data to predict value_error
X_train, X_test, y_train, y_test = \
                         train_test_split(X, y[:,0], test_size=0.33, random_state=state)

y_test, y_train = np.array(y_test), np.array(y_train)

clf = RandomForestRegressor(bootstrap= True, min_samples_leaf=2, n_estimators=1800, max_features='auto', min_samples_split=2, max_depth=None)
clf.fit(X_train, y_train)
# print(np.argmax(clf.feature_importances_))
pred = clf.predict(X_test)
print(explained_variance_score(pred, y_test))
print(r2_score(pred, y_test))

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
