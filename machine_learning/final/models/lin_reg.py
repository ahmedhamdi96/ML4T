from machine_learning.final.utils.dataset import bulid_TIs_dataset
from machine_learning.final.evaluation.metrics import evaluate
import numpy as np
import scipy.optimize as spo

'''computes and returns the root mean squared error

*x : a dynamic variable: (value, array, ...)
*y : a dynamic variable: (value, array, ...)
'''
def calculate_rmse(x, y):
    #squared error
    se = (x-y) ** 2
    #mean squared error
    mse = np.mean(se)
    #root mean squared error
    rmse = mse ** 0.5
    return rmse

'''given the fitted line coefficients and the dataset, this
function computes the rmse between the actual values and 
the predicted values of the linear regression

*coefficients : fitted line coefficients array
*data         : dataset containing the features and the output
'''
def error_fun(coefficients, data):
    price = coefficients[0]*data[:, 0]
    moment = coefficients[1]*data[:, 1]
    sma = coefficients[2]*data[:, 2]
    b_band = coefficients[3]*data[:, 3]
    std = coefficients[4]*data[:, 4]
    vroc = coefficients[5]*data[:, 5]
    constant = coefficients[6]
    predicted_values = price+moment+sma+b_band+std+vroc+constant
    actual_values = data[:, -1]
    rmse = calculate_rmse(predicted_values, actual_values)
    return rmse

'''given the data to be passed to the error fcn, this function 
computes an initial guess of the coefficients and uses SciPy's
minimize fcn and the error fcn to find the optimal coefficients

*data    : fitted line coefficients array
*err_fun : error function to be minimized by SciPy's minimizor
'''
def minimize_new_err_fun(data, err_fun):
    price = np.mean(data[:, 0])
    moment = np.mean(data[:, 1])
    sma = np.mean(data[:, 2])
    b_band = np.mean(data[:, 3])
    std = np.mean(data[:, 4])
    vroc = np.mean(data[:, 5])
    constant = 0
    coefficients_guess = [price, moment, sma, b_band, std, vroc, constant]
    result = spo.minimize(error_fun, coefficients_guess, args=(data, ), method="SLSQP", options= {'disp' : True})
    return result.x

def dataset_reshape(dataset, future_gap, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    print("Applying Future Gap...")
    X = X[:-future_gap]
    Y = Y[future_gap:]
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

def final_test_linreg(stock_symbol, start_date, end_date, window, future_gap):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for LinReg
    print("\n> reshaping the dataset for LinReg...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = dataset_reshape(ds_train, future_gap, None)
    X_test, Y_test = dataset_reshape(ds_test, future_gap, None)
    #fitting the training data
    print("\n> fitting the training data...")
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    training_set = np.concatenate((X_train, Y_train), axis=1)
    fitted_line_coefficients = minimize_new_err_fun(training_set, error_fun)
    print("Line Coefficients:", fitted_line_coefficients)
    #predictions
    price = fitted_line_coefficients[0]*X_test[:, 0]
    moment = fitted_line_coefficients[1]*X_test[:, 1]
    sma = fitted_line_coefficients[2]*X_test[:, 2]
    b_band = fitted_line_coefficients[3]*X_test[:, 3]
    std = fitted_line_coefficients[4]*X_test[:, 4]
    vroc = fitted_line_coefficients[5]*X_test[:, 5]
    constant = fitted_line_coefficients[6]
    predictions = price+moment+sma+b_band+std+vroc+constant
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