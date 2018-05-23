from utils.util import get_stock_data
import machine_learning.development.dataset_preprocessing as dpp
import machine_learning.development.linear_regression as lin_reg
from machine_learning.development.new_regression.new_dataset import compute_mape
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

def bulid_new_TIs_dataset(stock_symbol, start_date, end_date, window, normalize=True):
    cols = ["Date", "Adj Close", "Volume"]
    df = get_stock_data(stock_symbol, start_date, end_date, cols)
    df.rename(columns={"Adj Close" : 'price'}, inplace=True)
    df['momentum'] = dpp.compute_momentum_ratio(df['price'], window)
    df['sma'] = dpp.compute_sma_ratio(df['price'], window)
    df['bolinger_band'] = dpp.compute_bollinger_bands_ratio(df['price'], window)
    df['volatility'] = dpp.compute_volatility_ratio(df['price'], window)
    df['vroc'] = dpp.compute_vroc_ratio(df['Volume'], window)
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

def dataset_reshape(dataset, future_gap, split):
    print("Dataset Shape:", dataset.shape)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    print("Applying Future Gap...")
    X = X[:-future_gap]
    Y = Y[future_gap:]
    print("X Shape:", X.shape)
    print("Y Shape:", Y.shape)

    if split != None:
        print("Applying training, testing split...")
        split_index = int(split*X.shape[0])
        X_train = X[:split_index]
        X_test = X[split_index:]
        Y_train = Y[:split_index]
        Y_test = Y[split_index:]
        print("(X_train, Y_train, X_test, Y_test) Shapes:")
        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test
    
    return X, Y

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

def final_test_linreg(stock_symbol, start_date, end_date, window, future_gap):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_new_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_new_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for LinReg
    print("\n> reshaping the dataset for LinReg...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = dataset_reshape(ds_train, future_gap, None)
    X_test, Y_test = dataset_reshape(ds_test, future_gap, None)
    #fitting the training data
    print("\n> fitting the training data...")
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    training_set = np.concatenate((X_train, Y_train), axis=1)
    fitted_line_coefficients = lin_reg.minimize_new_err_fun(training_set, lin_reg.error_fun)
    print("Line Coefficients:", fitted_line_coefficients)
    #predictions
    price = fitted_line_coefficients[0]*X_test[:, 0]
    moment = fitted_line_coefficients[1]*X_test[:, 1]
    sma = fitted_line_coefficients[2]*X_test[:, 2]
    b_band = fitted_line_coefficients[3]*X_test[:, 3]
    std = fitted_line_coefficients[4]*X_test[:, 4]
    vroc = fitted_line_coefficients[5]*X_test[:, 5]
    constant = fitted_line_coefficients[4]
    predictions = price+moment+sma+b_band+std+vroc+constant
    #inverse-scaling
    print("\n> inverse-scaling the scaled values...")
    predictions = predictions.reshape((predictions.shape[0], 1))
    predictions_inv_scaled = scaler.inverse_transform(predictions)
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    Y_test_inv_scaled = scaler.inverse_transform(Y_test)
    #evaluation
    normalized_metrics, inv_normalized_metrics = evaluate(Y_test, predictions, 
                                                          Y_test_inv_scaled, predictions_inv_scaled)
    #grouping the actual prices and predictions
    print("\n> grouping the actual prices and predictions...")
    feature_cols = df_test.columns.tolist()
    feature_cols.remove("actual_price")
    df_test.drop(columns=feature_cols, inplace=True)
    df_test.rename(columns={"actual_price" : 'Actual'}, inplace=True)
    df_test = df_test.iloc[future_gap:]
    df_test['Actual'] = Y_test_inv_scaled
    df_test['Prediction'] = predictions_inv_scaled

    return normalized_metrics, inv_normalized_metrics, df_test