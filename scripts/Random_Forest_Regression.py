import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score

X = pd.read_csv('../data/cleaned_training_data_norm.csv')
y = pd.read_csv('../data/cleaned_training_data_labels_norm.csv')
y = np.array(y)[:,1:]

test_data = pd.read_csv('../data/cleaned_testing_data_norm.csv')

#create and train model
model_val_error = RandomForestRegressor(n_estimators=100)
model_val_error.fit(X, y[:, 0])
#print(clf.feature_importances_)
pred_val_error = model_val_error.predict(test_data)

model_train_error = RandomForestRegressor(n_estimators=100)
model_train_error.fit(X, y[:, 1])
#print(clf.feature_importances_)
pred_train_error = model_train_error.predict(test_data)

pred = np.stack((pred_val_error, pred_train_error), axis=1)
print(np.shape(pred))


# Put into csv file
# 2nd col: test_{0 -> 475}_val_error | test_{0-> 475}_train_error
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
pred_df.to_csv('/Users/vllgsbr2/Desktop/AutoML_Proj/data/submission.csv', index=False)
