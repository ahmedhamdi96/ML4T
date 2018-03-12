import read_and_plot_stocks
import matplotlib.pyplot as plt

def get_bollinger_bands(mean, std):
    upper_band = mean + (2*std)
    lower_band = mean - (2*std)
    return upper_band, lower_band

def main():
    start_date = "01/01/2017"
    end_date = "31/12/2017"
    symbols = ["SPY", "AAPL", "FB"]

    df = read_and_plot_stocks.get_data(symbols, start_date, end_date)
    print(df)

    spy = df["SPY"]
    ax = spy.plot(title = "Stock Prices", label="SPY")
    rm_SPY = spy.rolling(window=20).mean()
    std_SPY = spy.rolling(window=20).std()
    upper_band, lower_band = get_bollinger_bands(rm_SPY, std_SPY)

    rm_SPY.plot(label="Rolling Mean", ax=ax)
    upper_band.plot(label="Upper Bollinger Band", ax=ax)
    lower_band.plot(label="Lower Bollinger Band", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()