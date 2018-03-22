import numpy as np 
import scipy.optimize as spo 
import matplotlib.pyplot as plt 

def error_fun(coefficients, data):
    actual_values = data[:, 1]
    predicted_values = coefficients[0] * data[:, 0] + coefficients[1]
    return np.sum((actual_values - predicted_values) ** 2)

def minimize_err_fun(data, err_fun):
    coefficients_guess = [0, np.mean(data[:, 1])]
    result = spo.minimize(error_fun, coefficients_guess, args=(data, ), method="SLSQP", options= {'disp' : True})
    return result.x

def main():
    #Original Line
    original_line_coefficients = [6, 3]
    print("Original Line:", "y = {}x + {}".format(original_line_coefficients[0], original_line_coefficients[1]))
    x_points = np.linspace(0, 10, 50)
    y_points = original_line_coefficients[0] * x_points + original_line_coefficients[1]
    plt.plot(x_points, y_points, 'b-', label="Original Line")
    #Original Line after applying noise
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, y_points.shape)
    data_points = np.asarray([x_points, y_points+noise]).T
    plt.plot(data_points[:, 0], data_points[:, 1], 'go', label='Data Points')
    #Fitted Line
    fitted_line_coefficients = minimize_err_fun(data_points, error_fun)
    print("Fitted Line:", "y = {}x + {}".format(fitted_line_coefficients[0], fitted_line_coefficients[1]))
    plt.plot(data_points[:, 0], fitted_line_coefficients[0]*data_points[:, 0] + fitted_line_coefficients[1], 
            'r--', label='Fitted Line')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()