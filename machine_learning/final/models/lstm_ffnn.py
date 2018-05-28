from utils.util import get_stock_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from machine_learning.final.evaluation.metrics import evaluate
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam

def bulid_dataset(stock_symbol, start_date, end_date, window, time_steps, future_gap, normalize=True):
    cols = ["Date", "Adj Close"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    i = 1
    while i < time_steps:
        df['price'+str(i)] = df['price'].shift(-i)
        i = i+1
    df = df[:len(df)-time_steps+1]
    if time_steps == 1:
        column = 'price'
    else:
        column = 'price'+str(i-1)
    df['actual_price'] = df[column].shift(-future_gap)
    df = df[:-future_gap]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
        i = 1
        while i < time_steps:
            df['price'+str(i)] = scaler.fit_transform(df['price'+str(i)].values.reshape(-1,1))
            i = i+1
        df['actual_price'] = scaler.fit_transform(df['actual_price'].values.reshape(-1,1))
        
    print(df.head())
    print(df.tail())
    return df, scaler

def dataset_split(dataset, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

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

def final_test_lstm_ffnn(stock_symbol, start_date, end_date, window, time_steps, future_gap, neurons, 
                    drop_out, batch_size, epochs, validation_split, verbose, callbacks):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_dataset(stock_symbol, None, start_date, window, time_steps, future_gap)
    df_test, scaler = bulid_dataset(stock_symbol, start_date, end_date, window, time_steps, future_gap)
    #reshaping the dataset for LSTM FFNN
    print("\n> reshaping the dataset for LSTM FFNN...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = dataset_split(ds_train, None)
    X_test, Y_test = dataset_split(ds_test, None)
    #building the LSTM FFNN model
    print("\n> building the LSTM FFNN model...")
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
    df_test['Actual'] = Y_test_inv_scaled
    df_test['Prediction'] = predictions_inv_scaled

    return normalized_metrics, inv_normalized_metrics, df_test