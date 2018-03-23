''' this file contains functions that are used in most files
of this project, it contains utility functions to read and
plot stock historical data'''

import os
import pandas as pd
import matplotlib.pyplot as plt

'''this helper function redirects to the directory containing
the stock historical data

*symbol : stock symbol
*depth  : directory depth from the root
'''
def symbol_to_path(symbol, depth=1):
    base = os.path.dirname(__file__)

    while depth > 0:
        base = os.path.dirname(base)
        depth -= 1
    
    path =  os.path.join(base, "resources", "historical_data_2017", "{}.csv".format(symbol))
    return path

'''this function creates a dataframe of chosen stocks with 
dates as the index and the adjusted closing price of each
stock as the columns

*symbols        : stock symbol
*start_date     : start date of the dataframe's date index
*end_date       : end date of the dataframe's date index
*include_SPY    : boolean to indicate whether to include
                  S&P500 index stock                        
'''
def get_data(symbols, start_date, end_date, include_SPY=True):
    if include_SPY and "SPY" not in symbols:
        symbols.insert(0, "SPY")

    dates_index = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(index = dates_index)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",
                              parse_dates=True, usecols=["Date", "Adj Close"],
                              na_values="nan")
        df_temp = df_temp.rename(columns={"Adj Close" : symbol})
        df = df.join(df_temp, how="right")

    return df

'''this function plots a given dataframe

*dataframe      : dataframe to be plotted
*plot_title     : the plot title
*xlabel         : the horizontal axis label
*ylabel         : the vertical axis label
'''
def plot_data(dataframe, plot_title, xlabel, ylabel):
    ax = dataframe.plot(title=plot_title)
    ax.set_xlabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.show()

'''a tester function
'''
def main():
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["GOOG","AAPL","FB"]
    df = get_data(symbols, start_date, end_date)
    print(df)

    column_slicing = ['SPY', 'GOOG']
    dataframe_sliced = df.ix[:, column_slicing]
    plot_data(dataframe_sliced, "Selected Stock Prices", "Date", "Price")

'''to ensure running the tester function only when this file is run not imported
'''
if __name__ == "__main__":
    main()