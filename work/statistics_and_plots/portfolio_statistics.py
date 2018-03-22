from utils.util import get_data
import math

def compute_daily_portfolio_value(capital, symbols, start_date, end_date, allocations):
    df = get_data(symbols, start_date, end_date)
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
    return math.sqrt(252) * mean_daily_portfolio_return/std_daily_portfolio_return

def main():
    capital = 100000
    symbols = ["AAPL", "FB", "GOOG", "SPY"]
    allocations = [0.25, 0.25, 0.25, 0.25]
    start_date = "01/01/2017"
    end_date = "31/12/2017"

    #Daily Portfolio Value
    daily_portfolio_value = compute_daily_portfolio_value(capital, symbols, start_date, end_date, allocations)
    #print(daily_portfolio_value)

    #Daily Portfolio Return
    daily_portfolio_return = compute_daily_portfolio_return(daily_portfolio_value)
    #print(daily_portfolio_return)

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

if __name__ == "__main__":
    main()