from machine_learning.optimized_ffnn import ffnn
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def main(internal_eval=False):
    #building the dataset
    print("> building the dataset...")
    stock_symbol = '^GSPC'
    start_date = '1950-01-01'
    end_date = '2017-12-31'
    window = 5
    dataframe, scaler = ffnn.bulid_TIs_dataset(stock_symbol, start_date, end_date, window)

    #reshaping the dataset for FFNN
    print("\n> reshaping the dataset for FFNN...")
    dataset = dataframe.values
    future_gap = 5 #1 trading week
    split = 0.8 #80% of the dataset
    X_train, Y_train, X_test, Y_test = ffnn.ffnn_dataset_reshape(dataset, future_gap, split)

    #building the FFNN model
    print("\n> building the FFNN model...")
    features = X_train.shape[1]
    neurons = [256, 256, 16, 1]
    drop_out = 0.3
    verbose = 1
    model = ffnn.build_model(features, neurons, drop_out)

    #fitting the training data
    print("\n> fitting the training data...")
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                            patience=50, verbose=verbose, mode='auto')
    batch_size = 4096
    epochs = 200
    validation_split = 0.1
    history = ffnn.model_fit(model, X_train, Y_train, batch_size, epochs, validation_split,
                             verbose, [early_stopping_callback])

    #internal evaluation
    if internal_eval:
        print("\n> internal evaluation...")
        _, _ = ffnn.evaluate_model(model, X_train, Y_train, X_test, Y_test, verbose)

    #predictions
    predictions = model.predict(X_test)
    predictions = predictions.reshape((predictions.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))

    #evaluating the model on the normalized dataset
    rmse = (mean_squared_error(predictions, Y_test) ** 0.5)
    print('\nNormalized Test RMSE: %.3f' %(rmse))
    correlation = np.corrcoef(predictions.T, Y_test.T)
    print("Normalized Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(predictions, Y_test)
    print("Normalized Outsample r^2: %.3f"%(r2))

    #evaluating the model on the inverse-normalized dataset
    predictions_inv_scaled = scaler.inverse_transform(predictions)
    Y_test_inv_scaled = scaler.inverse_transform(Y_test)

    rmse = (mean_squared_error(predictions_inv_scaled, Y_test_inv_scaled) ** 0.5)
    print('\nInverse-Normalized Outsample RMSE: %.3f' %(rmse))
    correlation = np.corrcoef(predictions_inv_scaled.T, Y_test_inv_scaled.T)
    print("Inverse-Normalized Outsample Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(predictions_inv_scaled, Y_test_inv_scaled)
    print("Inverse-Normalized Outsample r^2: %.3f"%(r2))

    #plotting the results
    print("\n> plotting the results...")
    _, ax2 = plt.subplots()
    '''ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='best')
    ax1.grid(True)
    '''
    ax2.plot(range(len(predictions_inv_scaled)), predictions_inv_scaled, label='Prediction')
    ax2.plot(range(len(Y_test_inv_scaled)), Y_test_inv_scaled, label='Actual')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Price')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.show()

main()


#to be stored temporarily
'''#evaluating the model on the *dataset*
print("\n> evaluating the model on the *dataset*...")
predictions = model.predict(X_test)
Y_test = Y_test.reshape((Y_test.shape[0], 1))

predictions_inv_scaled = scaler.inverse_transform(predictions)
Y_test_inv_scaled = scaler.inverse_transform(Y_test)

rmse = (mean_squared_error(predictions_inv_scaled, Y_test_inv_scaled) ** 0.5)
print('Outsample RMSE: %.3f' %(rmse))
#correlation = np.corrcoef(predictions_inv_scaled, Y_test_inv_scaled)
#print("Outsample Correlation: %.3f"%(correlation[0, 1]))#
# '''