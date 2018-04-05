''' this file shows an implementation of linear regression to
predict stock prices one trading week in advance. SciPy's
minimize function is used to optimize the fitted linear line 
coefficients
'''
from utils.util import get_data
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from machine_learning.dataset_preprocessing import get_dataset_dataframe

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
    constant = coefficients[4]
    predicted_values = price+moment+sma+b_band+constant
    actual_values = data[:, -1]
    rmse = calculate_rmse(predicted_values, actual_values)
    return rmse

'''given the data to be passed to the error fcn, this function 
computes an initial guess of the coefficients and uses SciPy's
minimize fcn and the error fcn to find the optimal coefficients

*data    : fitted line coefficients array
*err_fun : error function to be minimized by SciPy's minimizor
'''
def minimize_err_fun(data, err_fun):
    price = np.mean(data[:, 0])
    moment = np.mean(data[:, 1])
    sma = np.mean(data[:, 2])
    b_band = np.mean(data[:, 3])
    constant = 0
    coefficients_guess = [price, moment, sma, b_band, constant]
    result = spo.minimize(error_fun, coefficients_guess, args=(data, ), method="SLSQP", options= {'disp' : True})
    return result.x

'''a normalization fcn

*values : values to be normalized
*mean   : mean of the values
*std    : standard deviation of the values
'''
def normalize(values, mean, std):
    return (values - mean) / std

'''an inverse-normalization fcn

*values : normalized values
*mean   : mean of the normalized values
*std    : standard deviation of the normalized values
'''
def inverse_normalize(normalized_values, mean, std):
    return (normalized_values * std) + mean

'''a tester function
'''
def main():
    #getting the preprocessed dataset dataframe
    dataset_df = get_dataset_dataframe()
    #dataset preparation
    dataset = dataset_df.values
    #dataset normalization
    '''mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    dataset_normalized = normalize(dataset, mean, std)
    '''
    #dataset splitting
    training_start_index = 0
    training_end_index = 503
    testing_start_index = 504
    testing_end_index = 755
    training_set = dataset[training_start_index:training_end_index+1, :]
    X_test = dataset[testing_start_index:testing_end_index+1, :-1]
    Y_test = dataset[testing_start_index:testing_end_index+1, -1]
    #training
    fitted_line_coefficients = minimize_err_fun(training_set, error_fun)
    print("Line Coefficients:", fitted_line_coefficients)
    #testing
    price = fitted_line_coefficients[0]*X_test[:, 0]
    moment = fitted_line_coefficients[1]*X_test[:, 1]
    sma = fitted_line_coefficients[2]*X_test[:, 2]
    b_band = fitted_line_coefficients[3]*X_test[:, 3]
    constant = fitted_line_coefficients[4]
    predicted_values = price+moment+sma+b_band+constant
    #evaluation
    rmse = calculate_rmse(predicted_values, Y_test)
    print('RMSE: %.3f' %(rmse))
    correlation = np.corrcoef(predicted_values, Y_test)
    print("Correlation: %.3f"%(correlation[0, 1]))
    #plots
    _, ax = plt.subplots()
    ax.plot(range(len(predicted_values)), predicted_values, label='Prediction')
    ax.plot(range(len(Y_test)), Y_test, label='Actual')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()

'''to ensure running the tester function only when this file is run, not imported
'''
if __name__ == "__main__":
    main()