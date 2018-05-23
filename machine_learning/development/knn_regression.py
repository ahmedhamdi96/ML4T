''' this file shows an implementation of kNN regression to
predict stock prices one trading week in advance
'''
from utils.util import get_data, plot_data
from machine_learning.development.dataset_preprocessing import get_dataset_dataframe
from machine_learning.development.linear_regression import calculate_rmse
import machine_learning.development.knn_wrapper as knn
import numpy as np
import matplotlib.pyplot as plt

'''a tester function
'''
def main():
    #getting the preprocessed dataset dataframe
    dataset_df = get_dataset_dataframe()
    #dataset preparation
    dataset = dataset_df.values
    #dataset splitting
    training_start_index = 0
    training_end_index = 503
    testing_start_index = 504
    testing_end_index = 755
    X_train = dataset[training_start_index:training_end_index+1, :-1]
    Y_train = dataset[training_start_index:training_end_index+1, -1]
    X_test = dataset[testing_start_index:testing_end_index+1, :-1]
    Y_test = dataset[testing_start_index:testing_end_index+1, -1]
    #kNN model
    model = knn.knn(3)
    #fitting the training data
    model.train(X_train, Y_train)
    #predictions
    predictions = model.query(X_test)
    #evaluation
    rmse = (calculate_rmse(predictions, Y_test) ** 0.5)
    print('Test RMSE: %.3f' %(rmse))
    correlation = np.corrcoef(predictions, Y_test)
    print("Correlation: %.3f"%(correlation[0, 1]))
    #plotting
    _, ax = plt.subplots()
    ax.plot(range(len(predictions)), predictions, label='Prediction')
    ax.plot(range(len(Y_test)), Y_test, label='Actual')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.show()

'''to ensure running the tester function only when this file is run, not imported
'''
if __name__ == "__main__":
    main()