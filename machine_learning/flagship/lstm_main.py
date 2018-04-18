from machine_learning.flagship import lstm
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def basic(print_flag, verbose, denormal_eval, plot):
    #building the dataset
    if print_flag:
        print("> building the dataset...")
    stock_symbol = '^GSPC'
    start_date = '1950-01-01'
    end_date = '2017-12-31'
    dataframe, scaler = lstm.bulid_dataset(stock_symbol, start_date, end_date)


    #reshaping the dataset for LSTM
    if print_flag:
        print("\n> reshaping the dataset for LSTM...")
    dataset = dataframe.values
    time_steps = 20 #1 trading month
    future_gap = 1 #1 trading day
    split = 0.9 #90% of the dataset
    X_train, Y_train, X_test, Y_test = lstm.lstm_dataset_reshape(dataset, time_steps, future_gap, split)


    #building the LSTM model
    if print_flag:
        print("\n> building the LSTM model...")
    features = X_train.shape[2]
    neurons = [128, 128, 32, 1]
    drop_out = 0.2
    model = lstm.build_model(time_steps, features, neurons, drop_out)


    #fitting the training data
    if print_flag:
        print("\n> fitting the training data...")
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                            patience=20, verbose=verbose, mode='auto')
    batch_size = 512
    epochs = 300
    validation_split = 0.1
    history = lstm.model_fit(model, X_train, Y_train, batch_size, epochs, validation_split,
                             verbose, [early_stopping_callback])


    #evaluating the model on the *normalized dataset*
    if print_flag:
        print("\n> evaluating the model on the *normalized dataset*...")
    train_mse, test_mse = lstm.evaluate_model(model, X_train, Y_train, X_test, Y_test, verbose)


    #evaluating the model on the *dataset*
    if denormal_eval:
        if print_flag:
            print("\n> evaluating the model on the *dataset*...")
        predictions = model.predict(X_test)
        Y_test = Y_test.reshape((Y_test.shape[0], 1))

        predictions_inv_scaled = scaler.inverse_transform(predictions)
        Y_test_inv_scaled = scaler.inverse_transform(Y_test)

        rmse = (mean_squared_error(predictions_inv_scaled, Y_test_inv_scaled) ** 0.5)
        if print_flag:
            print('Outsample RMSE: %.3f' %(rmse))
        '''correlation = np.corrcoef(predictions_inv_scaled, Y_test_inv_scaled)
        print("Outsample Correlation: %.3f"%(correlation[0, 1]))'''


    #plotting the results
    if plot:
        if print_flag:
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
    
    return train_mse, test_mse

def main():
    basic(True, 1, True, True)

main()