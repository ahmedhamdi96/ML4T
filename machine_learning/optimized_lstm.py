from utils.util import get_stock_data
import machine_learning.dataset_preprocessing as dpp
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

def build_model(time_steps, features, neurons, drop_out):
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(time_steps, features), return_sequences=True))
    model.add(Dropout(drop_out))
        
    model.add(LSTM(neurons[1], input_shape=(time_steps, features), return_sequences=False))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))

    model.compile(loss='mse',optimizer='adam')
    model.summary()
    return model

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    train_mse = model.evaluate(X_train, Y_train, verbose=1)
    print('Insample Testing: %.5f MSE (%.3f RMSE)' % (train_mse, (train_mse ** 0.5)))

    test_mse = model.evaluate(X_test, Y_test, verbose=1)
    print('Outsample Testing: %.5f MSE (%.3f RMSE)' % (test_mse, (test_mse ** 0.5)))


#building the dataset
print("> building the dataset...")
stock_symbol = '^GSPC'
start_date = '1950-01-01'
end_date = '2017-12-31'
window = 1
dataframe, scaler = bulid_dataset(stock_symbol, start_date, end_date, window)


#reshaping the dataset for LSTM
print("\n> reshaping the dataset for LSTM...")
dataset = dataframe.values
time_steps = 20 #1 trading month
future_gap = 1 #1 trading day
split = 0.9 #90% of the dataset
X_train, Y_train, X_test, Y_test = lstm_dataset_reshape(dataset, time_steps, future_gap, split)


#building the LSTM model
print("\n> building the LSTM model...")
features = X_train.shape[2]
neurons = [128, 128, 32, 1]
drop_out = 0.2
model = build_model(time_steps, features, neurons, drop_out)


#fitting the training data
print("\n> fitting the training data...")
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                        patience=20, verbose=1, mode='auto')
history = model.fit(
    X_train,
    Y_train,
    batch_size = 512,
    epochs = 300,
    validation_split = 0.1,
    verbose = 1,
    callbacks=[early_stopping_callback]
)


#evaluating the model on the *normalized dataset*
print("\n> evaluating the model on the *normalized dataset*...")
evaluate_model(model, X_train, Y_train, X_test, Y_test)


#evaluating the model on the *dataset*
print("\n> evaluating the model on the *dataset*...")
predictions = model.predict(X_test)
Y_test = Y_test.reshape((Y_test.shape[0], 1))

predictions_inv_scaled = scaler.inverse_transform(predictions)
Y_test_inv_scaled = scaler.inverse_transform(Y_test)

rmse = (mean_squared_error(predictions_inv_scaled, Y_test_inv_scaled) ** 0.5)
print('Outsample RMSE: %.3f' %(rmse))
'''correlation = np.corrcoef(predictions_inv_scaled, Y_test_inv_scaled)
print("Outsample Correlation: %.3f"%(correlation[0, 1]))'''


#plotting the results
print("\n> plotting the results...")
_, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(history.history['loss'], label='Training')
ax1.plot(history.history['val_loss'], label='Validation')
ax1.set_xlabel('Epoch #')
ax1.set_ylabel('Loss')
ax1.legend(loc='best')
ax1.grid(True)

ax2.plot(range(len(predictions_inv_scaled)), predictions_inv_scaled, label='Prediction')
ax2.plot(range(len(Y_test_inv_scaled)), Y_test_inv_scaled, label='Actual')
ax2.set_xlabel('Trading Day')
ax2.set_ylabel('Price')
ax2.legend(loc='best')
ax2.grid(True)

plt.show()