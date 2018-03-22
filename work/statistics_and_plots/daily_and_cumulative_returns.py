from utils.util import get_data, plot_data

def get_daily_returns(dataframe):
    #.values should be used to prevent pandas from auto matching indices
    daily_returns = dataframe.copy()
    daily_returns[1:] = (dataframe[1:]/dataframe[:-1].values) - 1
    #can also be done using pandas
    #daily_returns = (dataframe/dataframe.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns

def get_cumulative_returns(dataframe):
    cumulative_returns= (dataframe/dataframe.ix[0, :]) - 1
    return cumulative_returns

def main():
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["SPY", "FB"]

    df = get_data(symbols, start_date, end_date)
    print(df)
    plot_data(df, "Stock Prices", "Adj Close Price", "Date")

    daily_returns = get_daily_returns(df)
    plot_data(daily_returns, "Daily Returns", "Daily Return", "Date")

    cumulative_returns = get_cumulative_returns(df)
    plot_data(cumulative_returns, "Cumulative Returns", "Cumulative Return", "Date")

if __name__ == "__main__":
    main()