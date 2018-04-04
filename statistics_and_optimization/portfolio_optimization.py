from work.statistics_and_plots import portfolio_statistics
from utils.util import get_data, plot_data
import pandas as pd
import numpy as np 
import scipy.optimize as spo

def error_function(allocations, df_portfolio):
    #Daily Portfolio Value
    daily_portfolio_value = portfolio_statistics.compute_daily_portfolio_value(df_portfolio, 1, allocations)

    #Portfolio Statistics
    cummulative_portfolio_return, _, _, _ = portfolio_statistics.compute_portfolio_statistics(daily_portfolio_value)

    return -1*cummulative_portfolio_return

def compute_optimal_allocations(dataframe):
    guess = 1.0/dataframe.shape[1]
    allocations_guess = [guess] * dataframe.shape[1]
    bounds = [[0,1]] * dataframe.shape[1]
    constraints = {
                    'type':'eq', 
                    'fun': lambda allocations_guess : 1.0 - np.sum(allocations_guess)
                  }
    minimum = spo.minimize(error_function, allocations_guess, args=(dataframe, ),
                           method="SLSQP", bounds=bounds, constraints=constraints, options={'disp':True})
    return minimum.x

def optimization_main():
    symbols = ["AAPL", "FB", "GOOG", "SPY"]
    start_date = "01/01/2017"
    end_date = "31/12/2017"

    #Portfolio and SPY Dataframes
    df_portfolio = get_data(symbols, start_date, end_date)
    df_SPY = df_portfolio.ix[:, "SPY"]
    df_SPY = df_SPY/df_SPY.ix[0]

    #Optimized Allocations
    optimized_allocations = compute_optimal_allocations(df_portfolio)
    optimized_portfolio = portfolio_statistics.compute_daily_portfolio_value(df_portfolio, 100000, optimized_allocations)
    optimized_portfolio = optimized_portfolio/optimized_portfolio.ix[0]

    #Default Allocations
    default_allocations = [0.25, 0.25, 0.25,0.25]
    default_portfolio = portfolio_statistics.compute_daily_portfolio_value(df_portfolio, 100000, default_allocations)
    default_portfolio = default_portfolio/default_portfolio.ix[0]

    df_comparsion = pd.concat([optimized_portfolio, default_portfolio, df_SPY],
                              keys=["Optimized Portfolio","Default Portfolio","S&P500"], axis=1)
    
    portfolio_statistics.plot_data(df_comparsion, "Portfolio Optimization", "Date", "Price")

if __name__ == "__main__":
    optimization_main()