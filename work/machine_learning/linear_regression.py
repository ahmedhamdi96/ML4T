import os
import pandas as pd
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

def get_file_path(file_name, backtracking_depth, *directories):
    base = os.path.dirname(__file__)
    depth = backtracking_depth

    while depth > 0:
        base = os.path.dirname(base)
        depth -= 1

    for i in range(0, len(directories)):
        base =  os.path.join(base, directories[i])

    path =  os.path.join(base, file_name)
    return path

def symbol_to_path(symbol):
    base = os.path.dirname(__file__)
    depth = 2

    while depth > 0:
        base = os.path.dirname(base)
        depth -= 1
    
    path =  os.path.join(base, "resources", "historical_data_2017", "{}.csv".format(symbol))
    return path

def get_data(symbols, start_date, end_date):
    if "SPY" not in symbols:
        symbols.insert(0, "SPY")

    dates_index = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(index = dates_index)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",
        parse_dates=True, usecols=["Date", "Adj Close"], na_values="nan")
        df_temp = df_temp.rename(columns={"Adj Close" : symbol})
        df = df.join(df_temp, how="right")

    return df

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