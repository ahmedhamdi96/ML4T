from utils.util import get_stock_data
import machine_learning.dataset_preprocessing as dpp
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def bulid_dataset(stock_symbol, start_date, end_date, normalize=True):
    cols = ["Date", "Open", "Low", "High", "Adj Close"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['Open'] = scaler.fit_transform(df['Open'].values.reshape(-1,1))
        df['Low'] = scaler.fit_transform(df['Low'].values.reshape(-1,1))
        df['High'] = scaler.fit_transform(df['High'].values.reshape(-1,1))
        df['Adj Close'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))
    
    print(df.head())
    print(df.tail())
    return df, scaler

def bulid_TIs_dataset(stock_symbol, start_date, end_date, window, normalize=True):
    cols = ["Date", "Adj Close"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    df['momentum'] = dpp.compute_momentum_ratio(df['price'], window)
    df['sma'] = dpp.compute_sma_ratio(df['price'], window)
    df['bolinger_band'] = dpp.compute_bollinger_bands_ratio(df['price'], window)
    df['actual_price'] = df['price']
    df = df[window:]
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
        df['momentum'] = scaler.fit_transform(df['momentum'].values.reshape(-1,1))
        df['sma'] = scaler.fit_transform(df['sma'].values.reshape(-1,1))
        df['bolinger_band'] = scaler.fit_transform(df['bolinger_band'].values.reshape(-1,1))
        df['actual_price'] = scaler.fit_transform(df['actual_price'].values.reshape(-1,1))

    print(df.head())
    print(df.tail())
    return df, scaler

def lstm_dataset_reshape(dataset, time_steps, future_gap, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    X_sampled = []
    for i in range(X.shape[0] - time_steps + 1):
        X_sampled.append(X[i : i+time_steps])
    X_sampled = np.array(X_sampled)
    print("Sampled X Shape:", X_sampled.shape)

    future_gap_index = future_gap - 1
    X_sampled = X_sampled[:-future_gap]
    Y_sampled = Y[time_steps+future_gap_index: ]
    print("Applying Future Gap...")
    print("Sampled X Shape:", X_sampled.shape)
    print("Sampled Y Shape:", Y_sampled.shape)
    split_index = int(split*X_sampled.shape[0])
    X_train = X_sampled[:split_index]
    X_test = X_sampled[split_index:]
    Y_train = Y_sampled[:split_index]
    Y_test = Y_sampled[split_index:]
    print("(X_train, Y_train, X_test, Y_test) Shapes:")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def build_model(time_steps, features, neurons, drop_out, decay=0.0):
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(time_steps, features), return_sequences=True))
    model.add(Dropout(drop_out))
        
    model.add(LSTM(neurons[1], input_shape=(time_steps, features), return_sequences=False))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))

    adam = Adam(decay=decay)
    model.compile(loss='mse',optimizer=adam)
    model.summary()
    return model

def model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks):

    history = model.fit(
    X_train,
    Y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = validation_split,
    verbose = verbose,
    callbacks = callbacks
    )

    return history

def evaluate_model(model, X_train, Y_train, X_test, Y_test, verbose):
    train_mse = model.evaluate(X_train, Y_train, verbose=verbose)
    print('Insample Testing: %.5f MSE (%.3f RMSE)' % (train_mse, (train_mse ** 0.5)))

    test_mse = model.evaluate(X_test, Y_test, verbose=verbose)
    print('Outsample Testing: %.5f MSE (%.3f RMSE)' % (test_mse, (test_mse ** 0.5)))

    return train_mse, test_mse