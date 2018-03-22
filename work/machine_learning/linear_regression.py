from utils.util import get_data, plot_data
import os
import pandas as pd
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
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["SPY", "GOOG"]

    df = get_data(symbols, start_date, end_date)
    goog_price_train = df["GOOG"].ix[0:125]
    goog_price_test = df["GOOG"].ix[126:]

    ax = goog_price_train.plot(title = "Stock Prices", label="GOOG")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.show()

    goog_outstanding_shares = 349840000
    goog_net_income = 12662000
    goog_pe_ratio = df["GOOG"]/(goog_net_income/goog_outstanding_shares)
    goog_pe_ratio_train = goog_pe_ratio.ix[0:125]
    goog_pe_ratio_test = goog_pe_ratio.ix[126:]
    ax1 = goog_pe_ratio_train.plot(title = "PE Ratio", label="GOOG")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PE Ratio")
    ax1.legend(loc="upper left")
    plt.show()

    data_points = np.asarray([goog_pe_ratio_train, goog_price_train]).T
    fitted_line_coefficients = minimize_err_fun(data_points, error_fun)
    print("Fitted Line:", "y = {}x + {}".format(fitted_line_coefficients[0], fitted_line_coefficients[1]))
    plt.plot(data_points[:, 0], fitted_line_coefficients[0]*data_points[:, 0] + fitted_line_coefficients[1], 
                'r--', label='Fitted Line')
    plt.legend(loc="upper left")
    plt.show()

    #Actual and Predicted Prices Comparsion
    print(goog_price_test - (fitted_line_coefficients[0]*goog_pe_ratio_test + fitted_line_coefficients[1]) )
    
if __name__ == "__main__":
    main()