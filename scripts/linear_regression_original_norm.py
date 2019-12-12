import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

# Open original training data
data = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/train_OG.csv")
df = pd.DataFrame(data)

# Independent vars are all except what we're trying to predict; drop useless columns
X = df.drop(columns=['val_error', 'train_error', 'val_loss', 'train_loss', 'id', 'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu', 'init_params_std', 'init_params_l2'])

# What we're trying to predict
Y = df[['val_error', 'train_error']]

# Normalize training data
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

# Perform inear regression on training dataset
regr = linear_model.LinearRegression()
regr.fit(X,Y)

# Record learned coefficients & intercept
coef_valE = regr.coef_[0]
print(coef_valE)
coef_trainE = regr.coef_[1]
intercept_valE = regr.intercept_[0]
intercept_trainE = regr.intercept_[1]

# Prepare test data
test_data = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/test_OG.csv")
test_df = pd.DataFrame(test_data)
test = test_df.drop(columns=['id', 'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu', 'init_params_std', 'init_params_l2'])

# Normalize test data
test_x = test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
test_x_scaled = min_max_scaler.fit_transform(test_x)
test = pd.DataFrame(test_x_scaled)

pred = regr.predict(test)

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
        col2[i] = (pred[pred_row_idx][1])
        pred_row_idx = pred_row_idx+1
    # At even idx...val_error
    else:
        col1[i] = "test_"+ str(pred_row_idx) + "_val_error"
        col2[i] = (pred[pred_row_idx][0])

pred_dict = { 'id': col1, 
            'Predicted': col2 }
pred_df = pd.DataFrame(pred_dict, columns=['id', 'Predicted'])
pred_df.to_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/submission.csv', index=False)
