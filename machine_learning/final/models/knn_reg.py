from machine_learning.final.utils.dataset import bulid_TIs_dataset, dataset_split
import machine_learning.final.models.knn_wrapper as knn
from machine_learning.final.evaluation.metrics import evaluate

def final_test_knnreg(stock_symbol, start_date, end_date, window, future_gap, k):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for LinReg
    print("\n> reshaping the dataset for LinReg...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = dataset_split(ds_train, future_gap, None)
    X_test, Y_test = dataset_split(ds_test, future_gap, None)
    #kNN model
    model = knn.knn(k)
    #fitting the training data
    model.train(X_train, Y_train)
    #predictions
    predictions = model.query(X_test, normalize=False, addDiff=False)
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