import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from err import error_count, calc_diff
from RBF.rbflayer import RBFLayer, InitCentersRandom

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

# Size of dataset
dataset_sz = X.shape[0]

# load Trained Model
BKP = load_model('TICKER-BKP.h5')
RBF = load_model('TICKER-RBF.h5', custom_objects={'RBFLayer': RBFLayer})
RNN = load_model('TICKER-RNN.h5')


def eval(regressor, inputs):
    real_stock_price = np.array(X)
    predicted_stock_price = regressor.predict(inputs)
    
    # rebuild the Structure
    dataset_total = pd.DataFrame()
    dataset_total['real'] = real_stock_price
    dataset_total['predicted'] = predicted_stock_price
    
    # real test data price VS. predicted price
    predicted_stock_price = scaler.inverse_transform(dataset_total) 
    
    toler_rate = calc_diff(predicted_stock_price[:, 0], predicted_stock_price[:, 1])

    err_cnt = error_count(predicted_stock_price[:, 0], predicted_stock_price[:, 1], toler_treshold = 3.0)

    mse = mean_squared_error(predicted_stock_price[:, 0], predicted_stock_price[:, 1])
	
	mape = mean_absolute_percentage_error(all_prices[:, 0], all_prices[:, 1])

    return toler_rate, err_cnt, mse, mape



inputs = np.array(X)
inputs_rnn = np.reshape(inputs, (dataset_sz, 1, 1))

BKP_toler_rate, BKP_err_cnt, BKP_mse, BKP_mape = eval(BKP, inputs)
RBF_toler_rate, RBF_err_cnt, RBF_mse, RBF_mape = eval(RBF, inputs)
RNN_toler_rate, RNN_err_cnt, RNN_mse, RNN_mape = eval(RNN, inputs_rnn)