from utils.util import get_stock_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''technical indicators computation functions

*prices : adjusted closing stock prices
*window : rolling statistics window 
'''
#BEGIN
def compute_momentum_ratio(prices, window):
    #first window elements >> NA
    momentum_ratio = (prices/prices.shift(periods = 1)) - 1
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

def compute_volatility_ratio(prices, window):
    #first window-1 elements >> NA
    volatility_ratio = ((prices/prices.shift(periods = 1)) - 1).rolling(window = window).std()
    return volatility_ratio

def compute_vroc_ratio(volume, window):
    #Volume Rate of Change
    #first window-1 elements >> NA
    vroc_ratio = (volume/volume.shift(periods = window)) - 1
    return vroc_ratio

def compute_daily_return_volatility(prices, window):
    #first window-1 elements >> NA
    daily_return = (prices/prices.shift(periods= 1)) - 1
    volatility = daily_return.rolling(window=window).std()
    return volatility
#END

def bulid_TIs_dataset(stock_symbol, start_date, end_date, window, normalize=True):
    cols = ["Date", "Adj Close", "Volume"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    df['momentum'] = compute_momentum_ratio(df['price'], window)
    df['sma'] = compute_sma_ratio(df['price'], window)
    df['bolinger_band'] = compute_bollinger_bands_ratio(df['price'], window)
    df['volatility'] = compute_volatility_ratio(df['price'], window)
    df['vroc'] = compute_vroc_ratio(df['Volume'], window)
    df['actual_price'] = df['price']
    df.drop(columns=["Volume"], inplace=True)
    df = df[window:]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = None

    if normalize:        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
        df['momentum'] = scaler.fit_transform(df['momentum'].values.reshape(-1,1))
        df['sma'] = scaler.fit_transform(df['sma'].values.reshape(-1,1))
        df['bolinger_band'] = scaler.fit_transform(df['bolinger_band'].values.reshape(-1,1))
        df['volatility'] = scaler.fit_transform(df['volatility'].values.reshape(-1,1))
        df['vroc'] = scaler.fit_transform(df['vroc'].values.reshape(-1,1))
        df['actual_price'] = scaler.fit_transform(df['actual_price'].values.reshape(-1,1))
        
    print(df.head())
    print(df.tail())
    return df, scaler