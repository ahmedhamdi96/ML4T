import pandas as pd
from utils.util import get_data, plot_data

def compute_daily_portfolio_value(df, capital, allocations):
    #normalization
    normalized = df/df.ix[0, :]
    #allocation
    allocated = normalized*allocations
    #capital/position value
    pos_val = allocated*capital
    #port value
    port_val = pos_val.sum(axis=1)
    return port_val

def compute_daily_portfolio_return(daily_portfolio_value):
    return daily_portfolio_value[1:] / daily_portfolio_value[:-1].values - 1

def compute_cummulative_portfolio_return(daily_portfolio_value):
    return daily_portfolio_value[-1] / daily_portfolio_value[0] - 1

def compute_mean_daily_portfolio_return(daily_portfolio_return):
    return daily_portfolio_return.mean()

def compute_std_daily_portfolio_return(daily_portfolio_return):
    return daily_portfolio_return.std()

def compute_daily_sampled_sharpe_ratio(mean_daily_portfolio_return, std_daily_portfolio_return):
    return (252**0.5) * mean_daily_portfolio_return/std_daily_portfolio_return

def compute_portfolio_statistics(daily_portfolio_value):
    daily_portfolio_return =  compute_daily_portfolio_return(daily_portfolio_value)
    cummulative_portfolio_return = compute_cummulative_portfolio_return(daily_portfolio_value)
    mean_daily_portfolio_return =  compute_mean_daily_portfolio_return(daily_portfolio_return)
    std_daily_portfolio_return  = compute_std_daily_portfolio_return(daily_portfolio_return)
    daily_sampled_sharpe_ratio =  compute_daily_sampled_sharpe_ratio(mean_daily_portfolio_return, std_daily_portfolio_return)

    return cummulative_portfolio_return, mean_daily_portfolio_return, std_daily_portfolio_return, daily_sampled_sharpe_ratio

def main():
    capital = 100000
    symbols = ["AAPL", "FB", "GOOG", "SPY"]
    allocations = [0.25, 0.25, 0.25, 0.25]
    start_date = "01/01/2017"
    end_date = "31/12/2017"

    #Portfolio Dataframe
    df_portfolio = get_data(symbols, start_date, end_date)
    df_SPY = df_portfolio.ix[:, "SPY"]

    #Daily Portfolio Value
    daily_portfolio_value = compute_daily_portfolio_value(df_portfolio, capital, allocations)
    #print(daily_portfolio_value)

    #Daily Portfolio Return
    daily_portfolio_return = compute_daily_portfolio_return(daily_portfolio_value)

    #Cummulative Portfolio Return
    cummulative_portfolio_return = compute_cummulative_portfolio_return(daily_portfolio_value)
    print("Cummulative Portfolio Return:", cummulative_portfolio_return)

    #Daily Portfolio Return Mean
    mean_daily_portfolio_return = compute_mean_daily_portfolio_return(daily_portfolio_return)
    print("Daily Portfolio Return Mean:", mean_daily_portfolio_return)

    #Daily Portfolio Return Standard Deviation
    std_daily_portfolio_return = compute_std_daily_portfolio_return(daily_portfolio_return)
    print("Daily Portfolio Return Standard Deviation:", std_daily_portfolio_return)

    #Daily Sampled Sharpe Ratio
    daily_sampled_sharpe_ratio = compute_daily_sampled_sharpe_ratio(mean_daily_portfolio_return, std_daily_portfolio_return)
    print("Daily Sampled Sharpe Ratio:", daily_sampled_sharpe_ratio)

    #Comparing between the portfolio and S&P500
    daily_portfolio_value_normalized = daily_portfolio_value/daily_portfolio_value.ix[0]
    df_SPY_normalized = df_SPY/df_SPY.ix[0]
    df_comparsion = pd.concat([daily_portfolio_value_normalized, df_SPY_normalized], keys=["Portfolio", "SPY"], axis=1)
    plot_data(df_comparsion, "Portfolio 2017 Normalized Price", "Date", "Price")

if __name__ == "__main__":
    main()