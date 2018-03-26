import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
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
BKP = load_model('Trained_Model/TICKER-BKP.h5')
RBF = load_model('Trained_Model/TICKER-RBF.h5', custom_objects={'RBFLayer': RBFLayer})
RNN = load_model('Trained_Model/TICKER-RNN.h5')


def eval(regressor, inputs):
    real_stock_price = np.array(X)
    predicted_stock_price = regressor.predict(inputs)
    
    # rebuild the Structure
    dataset_total = pd.DataFrame()
    dataset_total['real'] = real_stock_price
    dataset_total['predicted'] = predicted_stock_price
    
    # real test data price VS. predicted price
    predicted_stock_price = scaler.inverse_transform(dataset_total) 
    
    toler_rate = np.zeros(dataset_sz)
    for i in range(0, dataset_sz):
        toler_rate[i] = abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1])
    
    toler_treshold = 3.0
    err_cnt = 0
    for i in range(0, dataset_sz):
        if abs(predicted_stock_price[i, 0] - predicted_stock_price[i, 1]) <= (toler_treshold/100) * predicted_stock_price[i, 0] :
            pass
        else:
            err_cnt +=1
    
    mse = 0.0
    for i in range(0, dataset_sz):
        mse += (predicted_stock_price[i, 0] - predicted_stock_price[i, 1])**2
    
    mse /= dataset_sz
    
    return toler_rate, err_cnt, mse



inputs = np.array(X)
BKP_toler_rate, BKP_err_cnt, BKP_mse = eval(BKP, inputs)

RBF_toler_rate, RBF_err_cnt, RBF_mse = eval(RBF, inputs)

inputs = np.reshape(inputs, (dataset_sz, 1, 1))
RNN_toler_rate, RNN_err_cnt, RNN_mse = eval(RNN, inputs)