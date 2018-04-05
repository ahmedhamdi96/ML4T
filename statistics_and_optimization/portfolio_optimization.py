''' this file finds the optimal portfolio allocation to maximize a 
chosen portfolio statistic
'''
from utils.util import get_data, plot_data
from statistics_and_optimization.portfolio_statistics import compute_daily_portfolio_value, compute_portfolio_statistics
import pandas as pd
import numpy as np 
import scipy.optimize as spo

'''this function returns a portfolio statistic to be maximized,
the value is multiplied by negative one, because it will be
passed to a minimizer in the compute_optimal_allocations fcn

*allocations    : given allocations to a portfolio
*df_portfolio   : the portfolio dataframe
'''
def portfolio_statistic(allocations, df_portfolio):
    #Daily Portfolio Value
    daily_portfolio_value = compute_daily_portfolio_value(df_portfolio, 1, allocations)

    #Portfolio Statistics
    cummulative_portfolio_return, _, _, _ = compute_portfolio_statistics(daily_portfolio_value)

    return -1*cummulative_portfolio_return

'''this function uses SciPy's minimizer and portfolio_statistic fcns
to minmize the negative portfolio statistic, and thus maximizing it
it returns the optimal allocation for maximizing the statistic

*dataframe  : the portfolio dataframe
'''
def compute_optimal_allocations(dataframe):
    guess = 1.0/dataframe.shape[1]
    allocations_guess = [guess] * dataframe.shape[1]
    bounds = [[0,1]] * dataframe.shape[1]
    constraints = {
                    'type':'eq', 
                    'fun': lambda allocations_guess : 1.0 - np.sum(allocations_guess)
                  }
    minimum = spo.minimize(portfolio_statistic, allocations_guess, args=(dataframe, ),
                           method="SLSQP", bounds=bounds, constraints=constraints,
                           options={'disp':True})
    return minimum.x

'''a tester function
'''
def main():
    symbols = ["AAPL", "FB", "GOOG", "SPY"]
    start_date = "01/01/2017"
    end_date = "31/12/2017"

    #Portfolio and SPY Dataframes
    df_portfolio = get_data(symbols, start_date, end_date)
    df_SPY = df_portfolio.ix[:, "SPY"]
    df_SPY = df_SPY/df_SPY.ix[0]

    #Optimized Allocations
    optimized_allocations = compute_optimal_allocations(df_portfolio)
    optimized_portfolio = compute_daily_portfolio_value(df_portfolio, 100000, optimized_allocations)
    optimized_portfolio = optimized_portfolio/optimized_portfolio.ix[0]

    #Default Allocations
    default_allocations = [0.25, 0.25, 0.25,0.25]
    default_portfolio = compute_daily_portfolio_value(df_portfolio, 100000, default_allocations)
    default_portfolio = default_portfolio/default_portfolio.ix[0]

    df_comparsion = pd.concat([optimized_portfolio, default_portfolio, df_SPY],
                              keys=["Optimized Portfolio","Default Portfolio","S&P500"], axis=1)
    
    plot_data(df_comparsion, "Portfolio Optimization", "Date", "Price")

'''to ensure running the tester function only when this file is run, not imported
'''
if __name__ == "__main__":
    main()