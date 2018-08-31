import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from rbflayer import RBFLayer, InitCentersRandom
from keras.models import load_model

#from err import mape
from abc import ABCMeta, abstractmethod



def mape(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



class Generalnn(object):
				"""
								Attributes:
								    csv_path: Path to the dataset file.
								    feature: extracted feature data
								    X, Y: splited Feature (input & output).
								    X_scaled, X_scaled: scaled Feature.
								    X_train, X_test, y_train, y_test: Scaled X, Y after spliting into test & training Dataset.
								    scaler
								    test_size
								    epochs, regressor, batch_size, epochs
				"""

				__metaclass__ = ABCMeta

				type = ""

				def __init__(self):
								pass


				def preprocess(self, csv_path, test_size=0.2):
								self.csv_path = csv_path
								self.test_size = test_size

								self.feature_extract()
								self.feature_scale()
								self.train_test_split()

								return self


				def train_test_split(self):
								self.X_train, self.X_test, self.y_train, self.y_test \
												= train_test_split(self.X_scaled, self.Y_scaled, test_size=self.test_size, random_state=0)


				def feature_scale(self):
								# Feature Scaling
								self.scaler = MinMaxScaler(feature_range=(0, 1))
								feature_scaled = self.scaler.fit_transform(self.feature)

								self.X_scaled = feature_scaled[:, 0]
								self.Y_scaled = feature_scaled[:, 1]


				def load_dataset(self):
								ds = pd.read_csv(self.csv_path)
								return ds


				def feature_extract(self):
								ds = self.load_dataset()
								self.feature = ds.iloc[:, [2, 5]].values

								self.X = ds.iloc[:, 2].values
								self.y = ds.iloc[:, 5].values



				def save_model(self, name):
								path =	'{}-{}.h5'.format(name, self.type)
								self.regressor.save(path)
								return path


				@abstractmethod
				def load_model(self, model_path=None):
								if self.regressor is None:
												self.regressor = load_model(model_path)
								else:
												return self.regressor



				# def retrain(self, batch_size=32, epochs=25):
				# 				self.regressor.compile(optimizer='adam', loss='mean_squared_error')
				# 				self.regressor.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs)



				@abstractmethod
				def train(self):
								pass


				@abstractmethod
				def test(self, model_path=None):
								pass


				@abstractmethod
				def predict(self, X, model_path=None):
								pass

