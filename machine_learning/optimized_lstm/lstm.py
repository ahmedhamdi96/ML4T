from utils.util import get_stock_data, plot_data
from machine_learning.testing.lag_metric import compute_lag_metric
import machine_learning.dataset_preprocessing as dpp
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def bulid_dataset(stock_symbol, start_date, end_date, normalize=True):
    cols = ["Date", "Open", "Low", "High", "Adj Close"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
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
    cols = ["Date", "Adj Close", "Volume"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    df['momentum'] = dpp.compute_momentum_ratio(df['price'], window)
    df['sma'] = dpp.compute_sma_ratio(df['price'], window)
    df['bolinger_band'] = dpp.compute_bollinger_bands_ratio(df['price'], window)
    df['volatility'] = dpp.compute_volatility_ratio(df['price'], window)
    df['vroc'] = dpp.compute_vroc_ratio(df['Volume'], window)
    df['actual_price'] = df['price']
    df.drop(columns=["Volume"], inplace=True)
    df = df[window:]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
        df['momentum'] = scaler.fit_transform(df['momentum'].values.reshape(-1,1))
        df['sma'] = scaler.fit_transform(df['sma'].values.reshape(-1,1))
        df['bolinger_band'] = scaler.fit_transform(df['bolinger_band'].values.reshape(-1,1))
        df['volatility'] = scaler.fit_transform(df['volatility'].values.reshape(-1,1))
        df['vroc'] = scaler.fit_transform(df['vroc'].values.reshape(-1,1))
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

    if split != None:
        split_index = int(split*X_sampled.shape[0])
        X_train = X_sampled[:split_index]
        X_test = X_sampled[split_index:]
        Y_train = Y_sampled[:split_index]
        Y_test = Y_sampled[split_index:]
        print("(X_train, Y_train, X_test, Y_test) Shapes:")
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    return X_sampled, Y_sampled

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

def test_lstm(stock_symbol, start_date, end_date, window, future_gap, time_steps,
              neurons, drop_out, batch_size, epochs, validation_split, verbose, callbacks, show_plot_flg):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for LSTM
    print("\n> reshaping the dataset for LSTM...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = lstm_dataset_reshape(ds_train, time_steps, future_gap, None)
    X_test, Y_test = lstm_dataset_reshape(ds_test, time_steps, future_gap, None)
    #building the LSTM model
    print("\n> building the LSTM model...")
    features = X_train.shape[2]
    model = build_model(time_steps, features, neurons, drop_out)
    #fitting the training data
    print("\n> fitting the training data...")
    model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks)
    #predictions
    print("\n> testing the model for predictions...")
    predictions = model.predict(X_test)
    #inverse-scaling
    print("\n> inverse-scaling the scaled values...")
    predictions = predictions.reshape((predictions.shape[0], 1))
    predictions_inv_scaled = scaler.inverse_transform(predictions)
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    Y_test_inv_scaled = scaler.inverse_transform(Y_test)
    #grouping the actual prices and predictions
    print("\n> grouping the actual prices and predictions...")
    feature_cols = df_test.columns.tolist()
    feature_cols.remove("actual_price")
    df_test.drop(columns=feature_cols, inplace=True)
    df_test.rename(columns={"actual_price" : 'Actual'}, inplace=True)
    df_test = df_test.iloc[time_steps+future_gap-1:]
    df_test['Actual'] = Y_test_inv_scaled
    df_test['Prediction'] = predictions_inv_scaled
    #ploting the forecast vs the actual
    print("\n> plotting the results...")
    lookup = 5
    lag_list = compute_lag_metric(df_test['Actual'], df_test['Prediction'], lookup, stock_symbol)

    df_test = df_test[:len(df_test)-lookup+1]
    plot_data(df_test, stock_symbol+" Price Forecast", "Date", "Price", show_plot=False)

    ax = df_test.plot(title=stock_symbol+" Price Forecast and PAL Overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    #sudden vs normal plot annotation
    ax.annotate('Normal Movement', xy=('2013-02-15', 40), xytext=('2013-03-05', 50), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
    ax.annotate('Sudden Change', xy=('2013-05-10', 55), xytext=('2013-03-05', 70), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
    ax1 = ax.twinx()
    ax1.scatter(df_test.index, lag_list, c='g')
    ax1.set_ylabel("PAL")

    if show_plot_flg:
        plt.show()