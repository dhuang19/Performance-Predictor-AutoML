import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import train_test_split


X_train = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/train_OG.csv')
Y_train = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/labels_OG.csv')
X_test = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/test_OG.csv')

Y_train_valE = Y_train.iloc[:, 0]
Y_train_trainE = Y_train.iloc[:, 1]

# Predict value error
rf_valE = RandomForestRegressor(bootstrap=True, min_samples_leaf=4, n_estimators=270, max_features='auto', min_samples_split=2, max_depth=100)
rf_valE.fit(X_train, Y_train_valE)
pred_valE = rf_valE.predict(X_test)

# Predict train error
rf_trainE = RandomForestRegressor(bootstrap=True, min_samples_leaf=2, n_estimators=1000, max_features='auto', min_samples_split=5, max_depth=60)
rf_trainE.fit(X_train, Y_train_trainE)
pred_trainE = rf_trainE.predict(X_test)

pred = np.stack((pred_valE, pred_trainE), axis=1)

# Put into csv file
col1 = np.empty(952, dtype=object)
col2 = np.empty(952, dtype=float)

pred_row_idx = 0
for i in range(952):
    #print("i : " +  str(i) + "  pred_row_idx : " + str(pred_row_idx))
    # At odd idx...train_error
    if ((i%2) != 0):
        col1[i] = "test_"+ str(pred_row_idx) + "_train_error"
        col2[i] = pred[pred_row_idx][1]
        pred_row_idx = pred_row_idx+1
    # At even idx...val_error
    else:
        col1[i] = "test_"+ str(pred_row_idx) + "_val_error"
        col2[i] = pred[pred_row_idx][0]

pred_dict = { 'id': col1,
            'Predicted': col2 }
pred_df = pd.DataFrame(pred_dict, columns=['id', 'Predicted'])
#print(pred_df)
pred_df.to_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/submission.csv', index=False)

# state = np.random.randint(0, 2838654)
# X_train, X_test, y_train, y_test = \
#                          train_test_split(X_train, Y_train, test_size=0.33, random_state=state)

# y_test, y_train = np.array(y_test), np.array(y_train)

# clf = RandomForestRegressor(n_estimators=10)
# clf.fit(X_train, y_train)
# # print(np.argmax(clf.feature_importances_))
# pred = clf.predict(X_test)
# print(explained_variance_score(pred, y_test))