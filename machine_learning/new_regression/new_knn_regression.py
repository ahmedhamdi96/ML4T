from machine_learning.new_regression import new_dataset as ds
from machine_learning.linear_regression import calculate_rmse
import machine_learning.knn_wrapper as knn
import numpy as np
import matplotlib.pyplot as plt

#building the dataset
print("> building the dataset...")
stock_symbol = '^GSPC'
start_date = '1950-01-01'
end_date = '2017-12-31'
window = 5
dataframe, scaler = ds.bulid_TIs_dataset(stock_symbol, start_date, end_date, window)

#reshaping the dataset
print("\n> reshaping the dataset...")
dataset = dataframe.values
future_gap = 5 #1 trading week
split = 0.8 #80% of the dataset
X_train, Y_train, X_test, Y_test = ds.dataset_reshape(dataset, future_gap, split)

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