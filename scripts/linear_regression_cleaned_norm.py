import pandas as pd
import numpy as np
from sklearn import linear_model

# Open cleaned training data
data = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_norm.csv")

# Independent vars are all cols
X = pd.DataFrame(data)

# What we're trying to predict
data_pred = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_training_data_labels_try.csv")
Y = pd.DataFrame(data_pred)

# Perform inear regression on training dataset
regr = linear_model.LinearRegression()
regr.fit(X,Y)
print(regr.score(X,Y))

# Prepare test data
test_data = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_testing_data_norm.csv")
test = pd.DataFrame(test_data)

pred = regr.predict(test)

# Put into csv file
# 2nd col: test_{0 -> 475}_val_error | test_{0-> 475}_train_error
col1 = np.empty(952, dtype=object)
col2 = np.empty(952, dtype=float)

pred_row_idx = 0
for i in range(952):
    #print("i : " +  str(i) + "  pred_row_idx : " + str(pred_row_idx))
    # At odd idx...val_error
    if ((i%2) != 0):
        col1[i] = "test_"+ str(pred_row_idx) + "_train_error"
        col2[i] = pred[pred_row_idx][1]
        pred_row_idx = pred_row_idx+1
    # At even idx...train_error
    else:
        col1[i] = "test_"+ str(pred_row_idx) + "_val_error"
        col2[i] = pred[pred_row_idx][0]

pred_dict = { 'id': col1, 
            'Predicted': col2 }
pred_df = pd.DataFrame(pred_dict, columns=['id', 'Predicted'])
pred_df.to_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/submission.csv', index=False)
