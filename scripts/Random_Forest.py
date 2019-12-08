import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import preprocessing

# X = pd.read_csv('../data/cleaned_training_data_norm.csv')
# y = pd.read_csv('../data/cleaned_training_data_labels_norm.csv')
# y = np.array(y)[:,1:]
#
# test_data = pd.read_csv('../data/cleaned_testing_data_norm.csv')
# print(X)



# Open original training data
data = pd.read_csv("/Users/vllgsbr2/Desktop/AutoML_Proj/data/train_OG.csv")
df = pd.DataFrame(data)

# Independent vars are all except what we're trying to predict; drop useless columns
X = df.drop(columns=['val_error', 'train_error', 'val_loss', 'train_loss', 'id',\
                     'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu',\
                     'init_params_std', 'init_params_l2'])


#normalize training data
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

# What we're trying to predict
y = df[['val_error', 'train_error']]

# Prepare test data
test_data = pd.read_csv("/Users/vllgsbr2/Desktop/AutoML_Proj/data/test_OG.csv")
test_df = pd.DataFrame(test_data)
test = test_df.drop(columns=['id', 'arch_and_hp', 'criterion', 'optimizer',\
                             'init_params_mu', 'init_params_std', 'init_params_l2'])

#normalize test data
x = test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
test = pd.DataFrame(x_scaled)

#reformat labels
est = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
est.fit(y)
y = est.transform(y).astype(np.int)

#create and train model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)
#print(clf.feature_importances_)
pred = est.inverse_transform(clf.predict(test))
#print(prediction)


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
