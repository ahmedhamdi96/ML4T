''' this file calculates and visualizes a company's stock
price bollinger bands, to be used as a trading strategy
'''
from utils.util import get_data, plot_data

'''given the rolling mean and std, calculate the upper and
lower bollinger bands

*mean : the rolling mean of a stock price
*std  : the rolling standard deviation of a stock price
'''
def get_bollinger_bands(mean, std):
    upper_band = mean + (2*std)
    lower_band = mean - (2*std)
    return upper_band, lower_band

'''a tester function
'''
def main():
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["FB"]
    stock_symbol = "FB"
    df = get_data(symbols, start_date, end_date, include_SPY=False)
    print(df.head())
    print(df.tail())
    
    window = 20
    rolling_mean = df[stock_symbol].rolling(window=window).mean()
    rolling_std = df[stock_symbol].rolling(window=window).std()
    df["Rolling Mean"] = rolling_mean
    df["Upper Bollinger Band"], df["Lower Bollinger Band"] = get_bollinger_bands(rolling_mean, rolling_std)
    plot_data(df, stock_symbol+" Bollinger Bands", "Date", "Price")

'''to ensure running the tester function only when this file is run not imported
'''
if __name__ == "__main__":
    main()