# ML4T
*Machine Learning for Trading*

## Project Overview
*GUC Bachelor Thesis Project*

This is the experimentation section of the thesis. 
* **Thesis Research Questions:**
  * *Can machine learning be used to predict future stock prices?*
  * *Can machine learning be used to generate profitable trading decisions?*

## Algorithms Evaluation (so far)
*Algorithm: (RMSE, Correlation)*
* <strong> Linear Regression: (3.328, 0.948)</strong>

```sh
python -m machine_learning.linear_regression
```
![Linear Regression](https://github.com/ahmedhamdi96/ML4T/blob/master/results/lin_reg.png)
* <strong> kNN Regression: (2.142, 0.905)</strong>

```sh
python -m machine_learning.knn_regression
```
![kNN Regression](https://github.com/ahmedhamdi96/ML4T/blob/master/results/knn.png)
* <strong> Keras Regression: (3.360, 0.947)</strong>

```sh
python -m machine_learning.keras_regression
```
![Keras Regression](https://github.com/ahmedhamdi96/ML4T/blob/master/results/keras_reg.png)
* <strong> Keras RNN LSTM: (3.405, 0.949)</strong>

```sh
python -m machine_learning.rnn_lstm
```
![Keras RNN LSTM](https://github.com/ahmedhamdi96/ML4T/blob/master/results/lstm.png)

## Software and Libraries
This project uses the following software and Python libraries:

* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/index.html)
* [SciPy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org)
* [Keras](https://keras.io/)
* [scikit-learn](http://scikit-learn.org/stable/)