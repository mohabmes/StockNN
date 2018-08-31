from generalnn import *

class stocknn(object):
				def __init__(self):
								pass


				class BKP(Generalnn):
								type = 'BKP'


								def __init__(self):
												super().__init__()
												print("backpropagation")


								def train(self, batch_size=32, epochs=50):
												regressor = Sequential()

												regressor.add(Dense(units=500, kernel_initializer='uniform', activation='relu', input_dim=1))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=500, kernel_initializer='uniform', activation='relu'))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=500, kernel_initializer='uniform', activation='relu'))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=500, kernel_initializer='uniform', activation='relu'))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

												regressor.compile(optimizer='adam', loss='mean_squared_error')
												regressor.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs)

												self.regressor = regressor

												return self


								def test(self, model_path=None):
												self.load_model(model_path)

												inputs = np.array(self.X_train)
												y_hat = self.regressor.predict(inputs)

												df = pd.DataFrame()
												df['X'] = inputs
												df['y_hat'] = y_hat

												predicted_price = self.scaler.inverse_transform(df)
												_mape = mape(predicted_price[:, 0], predicted_price[:, 1])
												_mse = mean_squared_error(predicted_price[:, 0], predicted_price[:, 1])

												return _mape, _mse


								def predict(self, X, model_path=None):
												self.load_model(model_path)

												x = np.array([X])
												yh = self.regressor.predict(x)

												x_yh = pd.DataFrame()
												x_yh['x'] = x
												x_yh['yh'] = yh

												x_yh = self.scaler.inverse_transform(x_yh)

												return x_yh[0]


				class RNN(Generalnn):
								type = 'RNN'


								def __init__(self):
												super().__init__()
												print("RNN")


								def train(self, batch_size=32, epochs=50):
												train_sz = self.X_train.shape[0]

												X_train = np.reshape(self.X_train, (train_sz, 1, 1))
												y_train = np.reshape(self.y_train, (train_sz, 1))

												regressor = Sequential()

												regressor.add(LSTM(units=500, return_sequences=True, input_shape=(X_train.shape[1], 1)))
												regressor.add(Dropout(.2))

												regressor.add(LSTM(units=500, return_sequences=True))
												regressor.add(Dropout(.2))

												regressor.add(LSTM(units=500, return_sequences=True))
												regressor.add(Dropout(.2))

												regressor.add(LSTM(units=500, return_sequences=False))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=1, activation='sigmoid'))

												regressor.compile(optimizer='adam', loss='mean_squared_error')

												regressor.fit(X_train, y_train, epochs=epochs, batch_size=32)

												self.regressor = regressor

												return self


								def test(self, model_path=None):
												self.load_model(model_path)

												inputs_ = np.array(self.X_test)
												inputs = np.reshape(inputs_, (self.X_test.shape[0], 1, 1))
												predicted_price = self.regressor.predict(inputs)

												df = pd.DataFrame()
												df['real'] = inputs_
												df['predicted'] = predicted_price

												predicted_price = self.scaler.inverse_transform(df)
												_mape = mape(predicted_price[:, 0], predicted_price[:, 1])
												_mse = mean_squared_error(predicted_price[:, 0], predicted_price[:, 1])

												return _mse, _mape


								def predict(self, X, model_path=None):
												self.load_model(model_path)

												x = np.array([X])
												x = np.reshape(x, (len(x), 1, 1))
												yh = self.regressor.predict(x)

												# rebuild the Structure
												x_yh = pd.DataFrame()
												x_yh['x'] = x[0][0]
												x_yh['yh'] = yh[0]

												# real test data price VS. predicted price
												x_yh = self.scaler.inverse_transform(x_yh)

												return x_yh[0]


				class RBF(Generalnn):
								type = 'RBF'


								def __init__(self):
												super().__init__()
												print("RBF")


								def train(self, batch_size=32, epochs=50):
												train_sz = self.X_train.shape[0]

												X_train = np.reshape(self.X_train, (train_sz, 1))
												y_train = np.reshape(self.y_train, (train_sz, 1))

												regressor = Sequential()

												regressor.add(RBFLayer(500, initializer=InitCentersRandom(X_train), betas=2.0, input_shape=(1,)))
												regressor.add(Dropout(.2))

												regressor.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

												regressor.compile(optimizer='adam', loss='mean_squared_error')

												regressor.fit(X_train, y_train, batch_size=32, epochs=epochs)

												self.regressor = regressor

												return self


								def load_model(self, model_path=None):
												if self.regressor is None:
																self.regressor = load_model(model_path, custom_objects={'RBFLayer': RBFLayer})


								def test(self, model_path=None):
												self.load_model(model_path)

												inputs = np.array(self.X_test)
												predicted_price = self.regressor.predict(inputs)

												df = pd.DataFrame()
												df['X'] = inputs
												df['y_hat'] = predicted_price

												predicted_price = self.scaler.inverse_transform(df)
												_mape = mape(predicted_price[:, 0], predicted_price[:, 1])
												_mse = mean_squared_error(predicted_price[:, 0], predicted_price[:, 1])

												return _mse, _mape


								def predict(self, X, model_path=None):
												self.load_model(model_path)

												x = np.array([X])
												yh = self.regressor.predict(x)

												x_yh = pd.DataFrame()
												x_yh['x'] = x
												x_yh['yh'] = yh

												x_yh = self.scaler.inverse_transform(x_yh)

												return x_yh[0]


obj = stocknn().RBF().preprocess('AAPL.csv', test_size=0.2).train(batch_size=32, epochs=1)
model = obj.save_model('AAPL')
mape, _ = obj.test(model)
_, pred = obj.predict(100)

#AAPL = stocknn().RNN()