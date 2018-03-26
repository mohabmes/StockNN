############ Data Preprocessing ############
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Training Set
ds = pd.read_csv('TICKER.csv')
dataset = ds.iloc[:, [2,5]].values

X = ds.iloc[:, 2].values
y = ds.iloc[:, 5].values

 
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
 
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Sizes of dataset, train_ds, test_ds
dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]
 

# reshape our data into 3 dimensions, [batch_size, timesteps, input_dim]
X_train = np.reshape(X_train, (train_sz, 1, 1))
y_train = np.reshape(y_train, (train_sz, 1))

 
############ Building the RNN ############
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
# Initializing the RNN
regressor = Sequential()
 
# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=500, return_sequences=True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(.2))
 
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=500, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 3rd layer with some drop out regularization
regressor.add(LSTM(units=500, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 4th layer with some drop out regularization
regressor.add(LSTM(units=500, return_sequences=False))
regressor.add(Dropout(.2))
 
# Output layer
regressor.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train :)
history = regressor.fit(X_train, y_train, epochs=50, batch_size=32)



############ Save & load Trained Model ############
# Save & load Trained Model
from keras.models import load_model

# Save Trained Model
regressor.save('TICKER-RNN.h5')

# deletes the existing model
del regressor

# load Trained Model
regressor = load_model('TICKER-RNN.h5')
 

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
inputs = np.reshape(inputs, (test_sz, 1, 1))
predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price

# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total) 

# Calc difference between real data price and predicted price
toler_rate = np.zeros(test_sz)

for i in range(0, test_sz):
    toler_rate[i] = abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1])

# tolerance threshold
toler_treshold = 5.0

# Wrong predicted count
err_cnt = 0
for i in range(0, test_sz):
    if abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1]) <= (toler_treshold/100) * predicted_stock_price[i, 0] :
        pass
    else:
        err_cnt +=1

# Calc MSE
mse = 0.0
for i in range(0, test_sz):
    mse += (predicted_stock_price[i, 0] - predicted_stock_price[i, 1])**2

mse /= test_sz

############ Visualizing the results ############
all_real_stock_price = np.array(y)
inputs = np.array(X)
inputs = np.reshape(inputs, (dataset_sz, 1, 1))
all_predicted_stock_price = regressor.predict(inputs)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = all_real_stock_price
dataset_test_total['predicted'] = all_predicted_stock_price

# real test data price VS. predicted price
stock_price_predicted_real = scaler.inverse_transform(dataset_test_total) 

# Visualising the results
plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.savefig('stock_price_predicted_real.png')
plt.show()


plt.plot(stock_price_predicted_real[:, 0], color = 'red', label = 'Real Stock Price')
plt.title('Stock Prices')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.savefig('Real_stock_price.png')
plt.show()


plt.plot(stock_price_predicted_real[:, 1], color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.savefig('Predicted_stock_price.png')
plt.show()