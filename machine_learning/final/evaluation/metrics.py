import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(Y_test, predictions, Y_test_inv_scaled, predictions_inv_scaled):
    rmse = (mean_squared_error(Y_test, predictions) ** 0.5)
    print('\nNormalized RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test, predictions) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test, predictions)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test, predictions)
    print('Normalized MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test.T, predictions.T)
    print("Normalized Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test, predictions)
    print("Normalized r^2: %.3f"%(r2))
    normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    #evaluating the model on the inverse-normalized dataset
    rmse = (mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5)
    print('\nInverse-Normalized Outsample RMSE: %.3f' %(rmse))
    nrmse = ((mean_squared_error(Y_test_inv_scaled, predictions_inv_scaled) ** 0.5))/np.mean(Y_test)
    print('Normalized NRMSE: %.3f' %(nrmse))
    mae = mean_absolute_error(Y_test_inv_scaled, predictions_inv_scaled)
    print('Normalized MAE: %.3f' %(mae))
    mape = compute_mape(Y_test_inv_scaled, predictions_inv_scaled)
    print('Inverse-Normalized Outsample MAPE: %.3f' %(mape))
    correlation = np.corrcoef(Y_test_inv_scaled.T, predictions_inv_scaled.T)
    print("Inverse-Normalized Outsample Correlation: %.3f"%(correlation[0, 1]))
    r2 = r2_score(Y_test_inv_scaled, predictions_inv_scaled)
    print("Inverse-Normalized Outsample r^2: %.3f"%(r2))
    inv_normalized_metrics = [rmse, nrmse, mae, mape, correlation[0, 1], r2]

    return normalized_metrics, inv_normalized_metrics

def compute_lag_metric(actual, prediction, lookup, symbol):
    diff_list = [None] * lookup
    lag_list = [None] * (len(actual)-lookup+1)

    for i in range(len(actual)-lookup+1):
        for j in range(lookup):
            diff_list[j] = abs(actual[i] - prediction[i+j])
        lag_list[i] = diff_list.index(min(diff_list))

    max_diff_count = [0] * lookup

    for i in range(len(lag_list)):
        max_diff_count[lag_list[i]] += 1

    _, ax = plt.subplots()
    ax.bar(range(len(max_diff_count)), max_diff_count, align='center')
    plt.sca(ax)
    plt.title(symbol+" Lag Test")
    ax.set_xlabel('Day Lag')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    _, ax1 = plt.subplots()
    index = actual[:len(actual)-lookup+1].index
    ax1.scatter(index, lag_list)
    plt.title(symbol+" Daily Lag Test")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Lag')
    ax1.grid(True)

    return lag_list