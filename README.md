# StockNN
Stock Prices Prediction Using Neural Network Models (Backpropagation, RNN LSTM, RBF) implemented in keras with Tensorflow backend to predict the daily closing price.

## Class Version Usage
```

snn = stocknn().RNN()

snn = snn.preprocess('AAPL.csv', test_size=0.2)
snn = snn.train(batch_size=32, epochs=50)

```
or
```
snn = stocknn().BKP().preprocess('AAPL.csv', test_size=0.5).train(batch_size=16, epochs=25)
```
```
model = snn.save_model('AAPL')
mape = snn.test(model)[0]
pred = snn.predict(100)[1]
```
#### StockNN Subclasses
```
[Subclasses]:
    stocknn().RNN()            Recurrent Neural Networks.
    stocknn().RBF()            Radial Basis Function Networks.
    stocknn().BKP()            Back-propagation Networks.
```


## Dataset
All datasets are obtained using [pystocklib](https://github.com/mohabmes/pystocklib).

## Requirement
- Keras
- Pandas
- numpy
- scikit-learn
- matplotlib

## Credit
- PetraVidnerova
