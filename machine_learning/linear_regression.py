from utils.util import get_data
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from machine_learning.dataset_preprocessing import get_dataset_dataframe

def calculate_rmse(x, y):
    #squared error
    se = (x-y) ** 2
    #mean squared error
    mse = np.mean(se)
    #root squared error
    rmse = mse ** 0.5
    return rmse

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

def minimize_err_fun(data, err_fun):
    price = np.mean(data[:, 0])
    moment = np.mean(data[:, 1])
    sma = np.mean(data[:, 2])
    b_band = np.mean(data[:, 3])
    constant = 0
    coefficients_guess = [price, moment, sma, b_band, constant]
    result = spo.minimize(error_fun, coefficients_guess, args=(data, ), method="SLSQP", options= {'disp' : True})
    return result.x

def normalize(values, mean, std):
    return (values - mean) / std

def denormalize(normalized_values, mean, std):
    return (normalized_values * std) + mean

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

if __name__ == "__main__":
    main()