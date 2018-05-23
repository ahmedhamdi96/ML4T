''' this file uses a keras feed-forward-NN to predict stock prices
one trading week in advance
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from machine_learning.development.dataset_preprocessing import get_dataset_dataframe

'''a tester function
'''
def main():
    #getting the preprocessed dataset dataframe
    dataset_df = get_dataset_dataframe()
    #dataset preparation
    dataset = dataset_df.values
    #dataset scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    #dataset splitting
    training_start_index = 0
    training_end_index = 503
    testing_start_index = 504
    testing_end_index = 755
    X_train = dataset[training_start_index:training_end_index+1, :-1]
    Y_train = dataset[training_start_index:training_end_index+1, -1]
    X_test = dataset[testing_start_index:testing_end_index+1, :-1]
    Y_test = dataset[testing_start_index:testing_end_index+1, -1]
    #Feed Forward NN model
    model = Sequential()
    model.add(Dense(20, input_dim=4, activation='relu', kernel_initializer='normal'))
    model.add(Dense(10, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='adam')
    #fitting the training data
    history = model.fit(X_train, Y_train, epochs=200, batch_size=int(X_train.shape[0]/8), 
                        validation_split=0.2, verbose=2, shuffle=False)
    #evaluating the testing data
    results = model.evaluate(X_test, Y_test)
    results_names = model.metrics_names
    print(results_names, ":", results)
    #predictions
    predictions_scaled = model.predict(X_test)
    test_dataset_scaled = np.concatenate((X_test, predictions_scaled), axis=1)
    test_dataset_unscaled = scaler.inverse_transform(test_dataset_scaled)
    predictions_unscaled = test_dataset_unscaled[:, -1]
    #actual values
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    test_dataset_scaled = np.concatenate((X_test, Y_test), axis=1)
    test_dataset_unscaled = scaler.inverse_transform(test_dataset_scaled)
    Y_test_unscaled = test_dataset_unscaled[:, -1]
    #evaluation
    rmse = (mean_squared_error(predictions_unscaled, Y_test_unscaled) ** 0.5)
    print('Test RMSE: %.3f' %(rmse))
    correlation = np.corrcoef(predictions_unscaled, Y_test_unscaled)
    print("Correlation: %.3f"%(correlation[0, 1]))
    #plotting
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

main()    