import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from err import error_count, calc_diff
from visual import plot

############ Data Preprocessing ############
# Importing the dataset
ds = pd.read_csv('TICKER.csv')
dataset = ds.iloc[:, [2,5]].values

X = ds.iloc[:, 2].values
y = ds.iloc[:, 5].values

# Feature Scaling
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Sizes of train_ds, test_ds
dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]

############ Building the ANN ############
# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1))
regressor.add(Dropout(.2))

# Adding the first hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the second hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the third hidden layer and Drop out Regularization
regressor.add(Dense(units = 500, kernel_initializer = 'uniform', activation = 'relu'))
regressor.add(Dropout(.2))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)


############ Save & load Trained Model ############

# Save Trained Model
regressor.save('TICKER-BKP.h5')

# deletes the existing model
del regressor

# load Trained Model
regressor = load_model('TICKER-BKP.h5')

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price
# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# count of Wrong predicted value after applying treshold
err_cnt = error_count(predicted_stock_price[:, 0], predicted_stock_price[:, 1], toler_treshold = 5.0)

# Calc difference between real data price and predicted price
diff_rate = calc_diff(predicted_stock_price[:, 0], predicted_stock_price[:, 1])

# MSE
mse = mean_squared_error(predicted_stock_price[:, 0], predicted_stock_price[:, 1])

############ Visualizing the results ############
inputs = np.array(X)

all_real_price = np.array(y)
all_predicted_price = regressor.predict(inputs)

# rebuild the Structure
dataset_pred_real = pd.DataFrame()
dataset_pred_real['real'] = all_real_price
dataset_pred_real['predicted'] = all_predicted_price

# real test data price VS. predicted price
all_prices = scaler.inverse_transform(dataset_pred_real)  

# Visualising the results
plot(predicted=all_prices[:, 0])
plot(real=all_prices[:, 1])
plot(predicted=all_prices[:, 0], real=all_prices[:, 1])