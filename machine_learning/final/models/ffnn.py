from machine_learning.final.utils.dataset import bulid_TIs_dataset, dataset_split
from machine_learning.final.evaluation.metrics import evaluate
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam

def build_model(features, neurons, drop_out, decay=0.0):
    model = Sequential()
    
    model.add(Dense(neurons[0], input_dim=features, activation='relu',))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[1], activation='relu'))
    model.add(Dropout(drop_out))
        
    model.add(Dense(neurons[2], activation='relu'))        
    model.add(Dense(neurons[3], activation='linear'))

    adam = Adam(decay=decay)
    model.compile(loss='mse',optimizer=adam)
    model.summary()
    return model

def model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks):

    history = model.fit(
        X_train,
        Y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = validation_split,
        verbose = verbose,
        callbacks = callbacks
    )

    return history

def final_test_ffnn(stock_symbol, start_date, end_date, window, future_gap, neurons, 
                    drop_out, batch_size, epochs, validation_split, verbose, callbacks):
    #building the dataset
    print("> building the dataset...")
    df_train, _ = bulid_TIs_dataset(stock_symbol, None, start_date, window)
    df_test, scaler = bulid_TIs_dataset(stock_symbol, start_date, end_date, window)
    #reshaping the dataset for FFNN
    print("\n> reshaping the dataset for FFNN...")
    ds_train = df_train.values
    ds_test = df_test.values
    X_train, Y_train = dataset_split(ds_train, future_gap, None)
    X_test, Y_test = dataset_split(ds_test, future_gap, None)
    #building the FFNN model
    print("\n> building the FFNN model...")
    features = X_train.shape[1]
    model = build_model(features, neurons, drop_out)
    #fitting the training data
    print("\n> fitting the training data...")
    model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks)
    #predictions
    print("\n> testing the model for predictions...")
    predictions = model.predict(X_test)
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