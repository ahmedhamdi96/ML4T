import math
import numpy as np

class knn:
    __k = 0
    __data = None

    def __init__(self, k):
        self.__k = k

    def train(self, data_x, data_y):
        self.__data = data_x.copy()
        self.__data['y_values'] = data_y.copy()
        self.__data['y_values'] = self.__data['y_values'].fillna(method='ffill')
        self.__data = self.__data.fillna(method='ffill')
        self.__data = self.__data.fillna(method='bfill')

    def query(self, data):
        data['predicted_prices'] = 0
        data['actual_prices_normed'] = data['actual_prices']/data['actual_prices'].ix[0, ['actual_prices']] - 1
        data = data.fillna(method='ffill')
        data = data.fillna(method='bfill')

        self.__data['distances'] = 0
        self.__data['actual_prices_normed'] = self.__data['actual_prices']/self.__data['actual_prices'].ix[0] - 1

        for i in range(0, data.shape[0]):

            bolinger_band_distance =  np.absolute(self.__data['bolinger_band'] - data['bolinger_band'].ix[i])
            momentum_distance = np.absolute(self.__data['momentum'] - data['momentum'].ix[i])
            volatility_distance = np.absolute(self.__data['volatility'] - data['volatility'].ix[i])
            actual_prices_normed_distance = np.absolute(self.__data['actual_prices_normed'] - data['actual_prices_normed'].ix[i])

            self.__data['distances'] = bolinger_band_distance+momentum_distance+volatility_distance+actual_prices_normed_distance

            data_sorted = self.__data.sort_values(['distances'])

            k_mean = np.mean(data_sorted['y_values'].ix[:self.__k])
            data['predicted_prices'].ix[i] = k_mean

        data['predicted_prices'] += data['actual_prices'].ix[0] - self.__data['actual_prices'].ix[0]
        return data['predicted_prices']