import numpy as np 
import scipy.optimize as spo
import matplotlib.pyplot as plt

def err_fun(coeff, data):
    actual_values = data[:, 1]
    predicted_values = np.polyval(coeff, data[:, 0])
    error = np.sum((actual_values - predicted_values) ** 2)
    return error

def fit_polynomial(data, err_fun, degree):
    coff = np.poly1d(np.ones(degree+1))
    min = spo.minimize(err_fun, coff, args=(data,), method="SLSQP", options={'disp':True})
    return np.poly1d(min.x)

def main():
    #Orginal Polynomial
    orginal_line_coeff = np.poly1d([1.5, -10, -5, 60, 50])
    x_points = np.linspace(-10, 10, 41)
    y_points = np.polyval(orginal_line_coeff, x_points)
    print("Orginal Polynomial: ")
    print(orginal_line_coeff)
    plt.plot(x_points, y_points, 'g--', label="Orginal Polynomial")

    #Orginal Line after applying noise
    noise_sigma = 10.0
    noise = np.random.normal(0, noise_sigma, x_points.shape)
    data_points = np.asarray([x_points, y_points+noise]).T
    plt.plot(data_points[:, 0], data_points[:, 1], 'bo', label="Data Points")

    #Fitted Polynomial
    fitted_line_coeff = fit_polynomial(data_points, err_fun, 4)
    print("Fitted Polynomial: ")
    print(fitted_line_coeff)
    y_points_fitted = np.polyval(fitted_line_coeff, x_points)
    plt.plot(x_points, y_points_fitted, 'r--', label="Fitted Polynomial")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()

