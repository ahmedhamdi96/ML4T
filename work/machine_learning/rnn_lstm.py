from utils.util import get_data, plot_data
import work.machine_learning.knn as knn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

def compute_momentum_ratio(prices, window):
    #first window elements >> NA
    momentum_ratio = (prices/prices.shift(periods = window)) - 1
    return momentum_ratio

def compute_sma_ratio(prices, window):
    #first window-1 elements >> NA
    sma_ratio = (prices / prices.rolling(window = window).mean()) - 1
    return sma_ratio

def compute_bollinger_bands_ratio(prices, window):
    #first window-1 elements >> NA
    bb_ratio = prices - prices.rolling(window = window).mean()
    bb_ratio = bb_ratio / (2 * prices.rolling(window = window).std())
    return bb_ratio

def compute_daily_return_volatility(prices, window):
    #first window-1 elements >> NA
    daily_return = (prices/prices.shift(periods= 1)) - 1
    volatility = daily_return.rolling(window=window).std()
    return volatility    

def normalize(prices):
    return prices - prices.mean()/prices.std()

def main():
    #importing stock data
    start_date = '17/12/2014' #include the last 2 weeks of 2014 to compensate calculations loss
    end_date = '31/12/2017'

    stock = 'IBM'
    stock_df = get_data([stock], start_date, end_date)
    date_range = pd.date_range(start_date, end_date)
    dataset_df = pd.DataFrame(index=date_range)

    #calculating technical indicators
    #1st week is lost in the preparation of the indicators
    #2nd week is lost to include the future gap
    future_gap = 5 #1 trading week
    dataset_df['price'] = stock_df[stock]
    dataset_df.dropna(subset=['price'], inplace=True)
    dataset_df['momentum'] = compute_momentum_ratio(stock_df[stock], future_gap)
    dataset_df['sma'] = compute_sma_ratio(stock_df[stock], future_gap)
    dataset_df['bolinger_band'] = compute_bollinger_bands_ratio(stock_df[stock], future_gap)
    #dataset_df['volatility'] = compute_daily_return_volatility(stock_df[stock], future_gap)
    dataset_df.dropna(subset=dataset_df.columns, inplace=True)
    dataset_df = dataset_df.shift(future_gap)
    shifted_columns_names = ['price(t-%d)' %(future_gap), 'moment(t-%d)' %(future_gap), 'sma(t-%d)' %(future_gap), 
                             'b_band(t-%d)' %(future_gap)]
    dataset_df.columns = shifted_columns_names
    dataset_df.dropna(subset=shifted_columns_names, inplace=True)
    dataset_df['price'] = stock_df[stock]

    #dataset preparation
    dataset = dataset_df.values
    #dataset scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    training_start_index = 0
    training_end_index = 503
    testing_start_index = 504
    testing_end_index = 755
    X_train = dataset[training_start_index:training_end_index+1, :-1]
    Y_train = dataset[training_start_index:training_end_index+1, -1]
    X_test = dataset[testing_start_index:testing_end_index+1, :-1]
    Y_test = dataset[testing_start_index:testing_end_index+1, -1]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    #reshaping the dataset for the LSTM RCC
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    (samples, timesteps, features) = X_train.shape

    #LSTM RNN model
    model = Sequential()
    model.add(LSTM(100, input_shape=(timesteps, features)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    #fitting the training data
    history = model.fit(X_train, Y_train, epochs=200, batch_size=int(samples/8), 
                        validation_split=0.2, verbose=2, shuffle=False)
    
    #evaluating the testing data
    results = model.evaluate(X_test, Y_test)
    results_names = model.metrics_names
    print(results_names, ":", results)

    # predictions
    predictions_scaled = model.predict(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    test_dataset_scaled = np.concatenate((X_test, predictions_scaled), axis=1)
    test_dataset_unscaled = scaler.inverse_transform(test_dataset_scaled)
    predictions_unscaled = test_dataset_unscaled[:, -1]

    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    test_dataset_scaled = np.concatenate((X_test, Y_test), axis=1)
    test_dataset_unscaled = scaler.inverse_transform(test_dataset_scaled)
    Y_test_unscaled = test_dataset_unscaled[:, -1]
    
    rmse = (mean_squared_error(predictions_unscaled, Y_test_unscaled) ** 0.5)
    print('Test RMSE: %.3f' % rmse)

    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='best')
    ax1.grid(True)

    ax2.plot(range(len(predictions_unscaled)), predictions_unscaled, label='Prediction')
    ax2.plot(range(len(Y_test_unscaled)), Y_test_unscaled, label='Actual')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Price')
    ax2.legend(loc='best')
    ax2.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()