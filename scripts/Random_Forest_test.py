import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


X = pd.read_csv('../data/cleaned_training_data_norm.csv')
y = pd.read_csv('../data/cleaned_training_data_labels_norm.csv')
y = np.array(y)[:,1:]

state = np.random.randint(0, 2838654)
X_train, X_test, y_train, y_test = \
                         train_test_split(X, y[:,1], test_size=0.33, random_state=state)

y_test, y_train = np.array(y_test), np.array(y_train)

# #call PCA on data
# pca_train = PCA(n_components=100)
# X_train = pca_train.fit_transform(X_train)
# pca_test = PCA(n_components=100)
# X_test = pca_test.fit_transform(X_test)
# X = pca.fit_transform(X)
# print(X.shape)
# print(pca.singular_values_)#explained_variance_ratio_)

#Random Forest
clf = RandomForestRegressor(n_estimators=100, max_depth=25)
clf.fit(X_train, y_train)
# print(np.argmax(clf.feature_importances_))
pred = clf.predict(X_test)
print(explained_variance_score(pred, y_test))

# #gradient boosting
# clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.3,
#                               max_depth=25, random_state=0).fit(X_train, y_train)
#
# pred = clf.predict(X_test)
#
# print(explained_variance_score(pred, y_test))

# state = np.random.randint(0, 2838654)
# #neural network
# clf = MLPRegressor(hidden_layer_sizes=(200, 200), tol=1e-3, max_iter=1000,\
#                                            random_state=state).fit(X_train, y_train)
# pred = clf.predict(X_test)
# print(explained_variance_score(pred, y_test))





























#
# test_data = pd.read_csv('../data/cleaned_testing_data_norm.csv')

# # Open original training data
# data = pd.read_csv("/Users/vllgsbr2/Desktop/AutoML_Proj/data/train_OG.csv")
# df = pd.DataFrame(data)
#
# # Independent vars are all except what we're trying to predict; drop useless columns
# X = df.drop(columns=['val_error', 'train_error', 'val_loss', 'train_loss', 'id',\
#                      'arch_and_hp', 'criterion', 'optimizer', 'init_params_mu',\
#                      'init_params_std', 'init_params_l2'])
#
# #normalize training data
# x = X.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# X_train = pd.DataFrame(x_scaled)
#
# # What we're trying to predict
# y_train = np.array(df[['val_error', 'train_error']])

# # Prepare test data
# test_data = pd.read_csv("/Users/vllgsbr2/Desktop/AutoML_Proj/data/test_OG.csv")
# test_df = pd.DataFrame(test_data)
# test = test_df.drop(columns=['id', 'arch_and_hp', 'criterion', 'optimizer',\
#                              'init_params_mu', 'init_params_std', 'init_params_l2'])
#
# #normalize test data
# x = test.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# X_test = pd.DataFrame(x_scaled)
