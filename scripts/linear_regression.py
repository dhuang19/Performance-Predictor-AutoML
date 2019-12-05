import pandas as pd
import numpy as np
from sklearn import linear_model

X = pd.read_csv('../data/cleaned_training_data_try.csv')
y = pd.read_csv('../data/cleaned_training_data_labels_try.csv')
y = np.array(y)[:,1:]

#print(y)


lm = linear_model.LinearRegression()
model = lm.fit(X,y)
#print R^2
#print(lm.score(X,y))


test_data = pd.read_csv('../data/cleaned_testing_data_try.csv')

predictions = np.array(lm.predict(test_data))

# print(np.shape(predictions))
# print(predictions)
#
# print(len(predictions))

predictions = predictions.flatten()
