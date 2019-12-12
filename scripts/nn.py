import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score

# Using updated cleaned data
# X = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/scripts/cleaned_training_data_norm.csv')
# Y = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/scripts/cleaned_training_data_labels_norm.csv')
# Y = np.array(Y)[:,1:]

# Y_ValError = Y[:, 0]
# Y_TrainError = Y[:, 1]

# X_test = pd.read_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/cleaned_testing_data_norm.csv')

# Using original data
X = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/train_OG.csv")
X_df = pd.DataFrame(X)
X_test = pd.read_csv("/home/diana/Desktop/Performance-Predictor-AutoML/data/test_OG.csv")
X_test =  pd.DataFrame(X_test).drop(columns=['id', 'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu', 'init_params_std', 'init_params_l2'])

X = X_df.drop(columns=['val_error', 'train_error', 'val_loss', 'train_loss', 'id', 'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu', 'init_params_std', 'init_params_l2'])
Y_ValError = X_df[['val_error']]
Y_TrainError = X_df[['train_error']]

# Feature scaling on training data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

max_val = 0
best_i_val = 1
best_j_val = 1

max_train = 0
best_i_train = 1
best_j_train = 1

for i in range(40, 91):
    for j in range(40, 91):
        print(str(i) + ", "+ str(j))
        # Run NN for val error
        mlp = MLPRegressor(hidden_layer_sizes=(i, j), max_iter=30000)
        mlp.fit(X, Y_ValError)
        pred_ValError = mlp.predict(X_test)
        # print(mlp.score(X, Y_ValError))

        temp = mlp.score(X, Y_ValError)
        if temp > max_val:
            max_val = temp 
            best_i_val = i
            best_j_val = j

        # Run NN for train error
        mlp = MLPRegressor(hidden_layer_sizes=(i, j), max_iter=30000)
        mlp.fit(X, Y_TrainError)
        pred_TrainError = mlp.predict(X_test)
        # print(mlp.score(X, Y_TrainError))

        temp = mlp.score(X, Y_TrainError)
        if temp > max_train:
            max_train = temp 
            best_i_train = i
            best_j_train = j

mlp = MLPRegressor(hidden_layer_sizes=(best_i_val, best_j_val), max_iter=30000)
mlp.fit(X, Y_ValError)
pred_ValError = mlp.predict(X_test)
print(mlp.score(X, Y_ValError))

# Run NN for train error
mlp = MLPRegressor(hidden_layer_sizes=(best_i_train, best_j_train), max_iter=30000)
mlp.fit(X, Y_TrainError)
pred_TrainError = mlp.predict(X_test)
print(mlp.score(X, Y_TrainError))

# mlp = MLPRegressor(hidden_layer_sizes=(80, 80), max_iter=30000)
# mlp.fit(X, Y_ValError)
# pred_ValError = mlp.predict(X_test)
# print(mlp.score(X, Y_ValError))

# # Run NN for train error
# mlp = MLPRegressor(hidden_layer_sizes=(80, 80), max_iter=30000)
# mlp.fit(X, Y_TrainError)
# pred_TrainError = mlp.predict(X_test)
# print(mlp.score(X, Y_TrainError))

pred = np.stack((pred_ValError, pred_TrainError), axis=1)

#Put into csv file
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
pred_df.to_csv('/home/diana/Desktop/Performance-Predictor-AutoML/data/submission.csv', index=False)
