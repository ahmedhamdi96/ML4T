from utils.util import get_stock_data
import machine_learning.dataset_preprocessing as dpp
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

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

def ffnn_dataset_reshape(dataset, future_gap, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    print("Applying Future Gap...")
    X = X[:-future_gap]
    Y = Y[future_gap:]

    print("Applying training, testing split...")
    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    print("(X_train, Y_train, X_test, Y_test) Shapes:")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def build_model(features, neurons, drop_out, decay=0.0):
    model = Sequential()
    
    model.add(Dense(neurons[0], input_dim=features, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[1], activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2], activation='relu', kernel_initializer="uniform"))        
    model.add(Dense(neurons[3], activation='linear', kernel_initializer="uniform"))

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