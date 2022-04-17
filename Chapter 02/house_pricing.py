import numpy as np
from scipy import rand
from sklearn import datasets 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle


# Load Housing data : 
data = datasets.load_boston()

# Shuffle the data :
X, y = shuffle(data.data, data.target, random_state = 7)

# Split the data into training and testing datasets :

num_training = int(0.8 * len(X))
X_train,y_train = X[:num_training], y[:num_training]
X_test,y_test = X[num_training:], y[num_training:]

# Create support vector regression Model : 
sv_regressor = SVR(kernel='linear', C = 1.0, epsilon = 0.1)

# Train Support Vector Regressor :
sv_regressor.fit(X_train, y_train)

# Evalute performane of support vector regression
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print('\n ### performance ###')
print('Mean squared error = ', round(mse,2))
print("Explain squared erro = ",round(mse, 2))
print('Explained variance score = ', round(evs,2))

# Test the regressor on test datapoint :
test_data = [3.7, 0, 18.4,1,0.87,5.95,91,2.5025,26,666,20.2,351.34,15.27]
print('\n Prediction price:', sv_regressor.predict([test_data])[0])