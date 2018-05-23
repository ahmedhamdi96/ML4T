from utils.util import get_stock_data
import machine_learning.development.dataset_preprocessing as dpp
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from machine_learning.development.new_regression.new_dataset import compute_mape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

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

def bulid_new_TIs_dataset(stock_symbol, start_date, end_date, window, normalize=True):
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

def ffnn_dataset_reshape(dataset, future_gap, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    print("Applying Future Gap...")
    X = X[:-future_gap]
    Y = Y[future_gap:]

    if split != None:
        print("Applying training, testing split...")
        split_index = int(split*X.shape[0])
        X_train = X[:split_index]
        X_test = X[split_index:]
        Y_train = Y[:split_index]
        Y_test = Y[split_index:]
        print("(X_train, Y_train, X_test, Y_test) Shapes:")
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test
    
    return X, Y

def build_model(features, neurons, drop_out, decay=0.0):
    model = Sequential()
    
    model.add(Dense(neurons[0], input_dim=features, activation='relu',))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[1], activation='relu'))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2], activation='relu'))        
    model.add(Dense(neurons[3], activation='linear'))

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

def evaluate(Y_test, predictions, Y_test_inv_scaled, predictions_inv_scaled):
    rmse = (mean_squared_error(Y_test, predictions) ** 0.5)
    print('\nNormalized RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test, predictions) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test, predictions)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test, predictions)
    print('Normalized MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test.T, predictions.T)
    print("Normalized Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test, predictions)
    print("Normalized r^2: %.3f"%(r2))
    normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    #evaluating the model on the inverse-normalized dataset
    rmse = (mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5)
    print('\nInverse-Normalized Outsample RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test_inv_scaled, predictions_inv_scaled)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test_inv_scaled, predictions_inv_scaled)
    print('Inverse-Normalized Outsample MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test_inv_scaled.T, predictions_inv_scaled.T)
    print("Inverse-Normalized Outsample Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test_inv_scaled, predictions_inv_scaled)
    print("Inverse-Normalized Outsample r^2: %.3f"%(r2))
    inv_normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    return normalized_metrics, inv_normalized_metrics

def final_test_ffnn(stock_symbol, start_date, end_date, window, future_gap, neurons, 
                    drop_out, batch_size, epochs, validation_split, verbose, callbacks):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_new_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_new_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for FFNN
    print("\n> reshaping the dataset for FFNN...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = ffnn_dataset_reshape(ds_train, future_gap, None)
    X_test, Y_test = ffnn_dataset_reshape(ds_test, future_gap, None)
    #building the FFNN model
    print("\n> building the FFNN model...")
    features = X_train.shape[1]
    model = build_model(features, neurons, drop_out)
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
    #evaluation
    normalized_metrics, inv_normalized_metrics = evaluate(Y_test, predictions, 
                                                          Y_test_inv_scaled, predictions_inv_scaled)
    #grouping the actual prices and predictions
    print("\n> grouping the actual prices and predictions...")
    feature_cols = df_test.columns.tolist()
    feature_cols.remove("actual_price")
    df_test.drop(columns=feature_cols, inplace=True)
    df_test.rename(columns={"actual_price" : 'Actual'}, inplace=True)
    df_test = df_test.iloc[future_gap:]
    df_test['Actual'] = Y_test_inv_scaled
    df_test['Prediction'] = predictions_inv_scaled

    return normalized_metrics, inv_normalized_metrics, df_test