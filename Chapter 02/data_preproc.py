import numpy as np
from sklearn import preprocessing


input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]
                      ])

# Binary data 
# Explanation if each scalar sup of treshold put 1 else put 0

data_binarize = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print('\n Binarize data : \n',data_binarize)

# Print mean and standart deviation :
print('\n Before:')
print('Mean = ', input_data.mean(axis = 0))
print('Std deviation =', input_data.std(axis = 0))

# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print('\n Min max scaled data : ', data_scaled_minmax)

# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm = 'l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

print('\n L1 normalized data : \n', data_normalized_l1)
print('\nL2 normalized data: \n', data_normalized_l2)