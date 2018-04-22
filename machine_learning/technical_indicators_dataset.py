'''This file constructs a dataset to be used by the ML algorithms.
The dataset consists of the past price and technical indicators as
features, and the price as the output. The dataset is indexed by
date, a row entry contains the price and techincal indicators of
some day prior to the date index, and the price is the actual 
price of the stock at the date marked by the index.
'''
from utils.util import get_stock_data
import numpy as np
import pandas as pd
import talib as ta

'''technical indicators computation functions

*prices : adjusted closing stock prices
*window : rolling statistics window 
'''
#BEGIN
def compute_momentum_ratio(prices, window):
    #first window elements >> NA
    momentum_ratio = (prices/prices.shift(periods = window)) - 1
    return momentum_ratio

def compute_sma_ratio(prices, window):
    #Simple Moving Average
    #first window-1 elements >> NA
    sma_ratio = (prices / prices.rolling(window = window).mean()) - 1
    return sma_ratio

def compute_bollinger_bands_ratio(prices, window):
    #first window-1 elements >> NA
    bb_ratio = prices - prices.rolling(window = window).mean()
    bb_ratio = bb_ratio / (2 * prices.rolling(window = window).std())
    return bb_ratio

def compute_daily_return_volatility(prices, window):
    #first window-1 elements >> NA
    daily_return = (prices/prices.shift(periods= 1)) - 1
    volatility = daily_return.rolling(window=window).std()
    return volatility
#END

'''dataset constructor function

*start_date : start date for the entire dataset (training and testing)
*end_date   : end date for the entire dataset (training and testing)
*stock      : stock label to be used in the dataset
'''
def get_dataset_dataframe(start_date='17/12/2014', end_date = '31/12/2017', stock='IBM'):
    #importing stock data
    columns = ["Date", "Adj Close", "High", "Low", "Volume"]
    stock_df = get_stock_data(stock, start_date, end_date, columns=columns) 
    date_range = pd.date_range(start_date, end_date)
    dataset_df = pd.DataFrame(index=date_range)
    #calculating technical indicators
    #make sure to include the last 2 weeks of 2014 to compensate calculations loss
    #1st week is lost in the preparation of the indicators
    #2nd week is lost to include the future gap
    future_gap = 5 #1 trading week
    dataset_df['price'] = stock_df["Adj Close"]
    dataset_df.dropna(subset=['price'], inplace=True)
    dataset_df['momentum'] = compute_momentum_ratio(stock_df["Adj Close"], future_gap)
    dataset_df['sma'] = compute_sma_ratio(stock_df["Adj Close"], future_gap)
    dataset_df['bolinger_band'] = compute_bollinger_bands_ratio(stock_df["Adj Close"], future_gap)
    dataset_df['sar'] = ta.SAR(stock_df["High"], stock_df["Low"])
    dataset_df['rsi'] = ta.RSI(stock_df["Adj Close"], timeperiod=future_gap)
    dataset_df['obv'] = ta.OBV(stock_df["Adj Close"], stock_df["Volume"])
    dataset_df['adosc'] = ta.ADOSC(stock_df["High"], stock_df["Low"], stock_df["Adj Close"], stock_df["Volume"],
                                   fastperiod=2, slowperiod=3)
    dataset_df['macd'], _, _ = ta.MACD(stock_df["Adj Close"], fastperiod=2, slowperiod=3, signalperiod=3)
    dataset_df['slowk '], dataset_df['slowd'] = ta.STOCH(stock_df["High"], stock_df["Low"], stock_df["Adj Close"],
                                                         fastk_period=3, slowk_period=2, slowd_period=3)
    dataset_df['cci'] = ta.CCI(stock_df["High"], stock_df["Low"], stock_df["Adj Close"], timeperiod=future_gap)
    dataset_df['volatility'] = compute_daily_return_volatility(stock_df["Adj Close"], future_gap)
    dataset_df.dropna(subset=dataset_df.columns, inplace=True)
    dataset_df = dataset_df.shift(future_gap)
    shifted_columns_names = ['price(t-%d)' %(future_gap), 'moment(t-%d)' %(future_gap), 'sma(t-%d)' %(future_gap), 
                             'b_band(t-%d)' %(future_gap), 'sar(t-%d)' %(future_gap), 'rsi(t-%d)' %(future_gap),
                             'obv(t-%d)' %(future_gap), 'adosc(t-%d)' %(future_gap), 'macd(t-%d)' %(future_gap),
                             'slowk(t-%d)' %(future_gap), 'slowd(t-%d)' %(future_gap), 'cci(t-%d)' %(future_gap),
                             'volatility(t-%d)' %(future_gap)]
    dataset_df.columns = shifted_columns_names
    dataset_df.dropna(subset=shifted_columns_names, inplace=True)
    dataset_df['price'] = stock_df["Adj Close"]

    return dataset_df