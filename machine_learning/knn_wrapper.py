''' this file contains an implementation of kNN regression
'''
import numpy as np

'''kNN wrapper class

*k       : k nearest neighbors to be considered
*dataset : training dataset including the features and the output
'''
class knn:
    __k = 0
    __dataset = None

    '''constructor function

    *k       : k nearest neighbors to be considered
    '''
    def __init__(self, k):
        self.__k = k

    '''training function

    *data_x : training dataset features
    *data_y : training dataset output
    '''
    def train(self, data_x, data_y):
        data_y_reshaped = data_y.reshape((data_y.shape[0], 1))
        self.__dataset = np.concatenate((data_x, data_y_reshaped), axis=1)

    '''querying/evaluating function

    *features : test dataset features
    '''
    def query(self, features):
        dataset_price_normed = (self.__dataset[:, 0]/self.__dataset[0, 0]) - 1
        features_price_normed = (features[:, 0]/features[0, 0]) - 1
        
        cumm_difference = np.zeros(features.shape[0])
        predicted_price = np.zeros(features.shape[0])

        for i in range(0, features.shape[0]):

            price_normed_difference = np.absolute(dataset_price_normed - features_price_normed[i])
            moment_difference = np.absolute(self.__dataset[:, 1] - features[i, 1])
            sma_difference = np.absolute(self.__dataset[:, 2] - features[i, 2])
            b_band_difference =  np.absolute(self.__dataset[:, 3] - features[i, 3])
            
            cumm_difference = price_normed_difference + moment_difference + sma_difference + b_band_difference
            difference_op = np.asarray([cumm_difference, self.__dataset[:, -1]]).T
            sorting_index = np.argsort(difference_op[:, 0])
            difference_sorted = difference_op[sorting_index]

            k_mean = np.mean(difference_sorted[:self.__k, 1])
            predicted_price[i] = k_mean

        predicted_price += features[0, 0] - self.__dataset[0, 0]
        return predicted_price