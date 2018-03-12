import os
import pandas as pd
import matplotlib.pyplot as plt

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
        '''df = df.join(df_temp)
        if symbol == "SPY":
            df = df.dropna(subset=["SPY"])'''
    
    #normalized
    #df/df.ix[0]
    return df

def plot_data(dataframe, plot_title, xlabel, ylabel):
    ax = dataframe.plot(title=plot_title)
    ax.set_xlabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.legend(loc="upper left")
    plt.show()

def main():
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["GOOG","AAPL","FB"]
    df = get_data(symbols, start_date, end_date)
    print(df)

    column_slicing = ['SPY', 'GOOG']
    dataframe_sliced = df.ix[:, column_slicing]
    plot_data(dataframe_sliced, "Selected Stock Prices", "Date", "Price")

if __name__ == "__main__":
    main()