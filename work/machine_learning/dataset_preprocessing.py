from utils.util import get_data, plot_data
import work.machine_learning.knn as knn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

def compute_momentum_ratio(prices, window):
    #first window elements >> NA
    momentum_ratio = (prices/prices.shift(periods = window)) - 1
    return momentum_ratio

def compute_sma_ratio(prices, window):
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

def normalize(prices):
    return prices - prices.mean()/prices.std()

def get_dataset_dataframe(start_date='17/12/2014', end_date = '31/12/2017', stock='IBM'):
    #importing stock data
    stock_df = get_data([stock], start_date, end_date) 
    date_range = pd.date_range(start_date, end_date)
    dataset_df = pd.DataFrame(index=date_range)

    #calculating technical indicators
    #make sure include the last 2 weeks of 2014 to compensate calculations loss
    #1st week is lost in the preparation of the indicators
    #2nd week is lost to include the future gap
    future_gap = 5 #1 trading week
    dataset_df['price'] = stock_df[stock]
    dataset_df.dropna(subset=['price'], inplace=True)
    dataset_df['momentum'] = compute_momentum_ratio(stock_df[stock], future_gap)
    dataset_df['sma'] = compute_sma_ratio(stock_df[stock], future_gap)
    dataset_df['bolinger_band'] = compute_bollinger_bands_ratio(stock_df[stock], future_gap)
    #dataset_df['volatility'] = compute_daily_return_volatility(stock_df[stock], future_gap)
    dataset_df.dropna(subset=dataset_df.columns, inplace=True)
    dataset_df = dataset_df.shift(future_gap)
    shifted_columns_names = ['price(t-%d)' %(future_gap), 'moment(t-%d)' %(future_gap), 'sma(t-%d)' %(future_gap), 
                             'b_band(t-%d)' %(future_gap)]
    dataset_df.columns = shifted_columns_names
    dataset_df.dropna(subset=shifted_columns_names, inplace=True)
    dataset_df['price'] = stock_df[stock]

    return dataset_df