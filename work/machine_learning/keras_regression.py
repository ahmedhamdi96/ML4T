from utils.util import get_data, plot_data
import work.machine_learning.knn as knn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from work.machine_learning.dataset_preprocessing import get_dataset_dataframe

def main():
    dataset_df = get_dataset_dataframe()
    #dataset preparation
    dataset = dataset_df.values
    #dataset scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    training_start_index = 0
    training_end_index = 503
    testing_start_index = 504
    testing_end_index = 755
    X_train = dataset[training_start_index:training_end_index+1, :-1]
    Y_train = dataset[training_start_index:training_end_index+1, -1]
    X_test = dataset[testing_start_index:testing_end_index+1, :-1]
    Y_test = dataset[testing_start_index:testing_end_index+1, -1]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

main()    