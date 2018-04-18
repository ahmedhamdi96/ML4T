from machine_learning.flagship import lstm
from keras.callbacks import EarlyStopping

def evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                  neurons, batch_size, epochs, validation_split, verbose, decay=0.0):

    dataframe, _ = lstm.bulid_dataset(stock, start_date, end_date)
    dataset = dataframe.values
    X_train, Y_train, X_test, Y_test = lstm.lstm_dataset_reshape(dataset, time_steps, future_gap, split)
    features = X_train.shape[2]
    model = lstm.build_model(time_steps, features, neurons, dropout, decay)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                            patience=20, verbose=verbose, mode='auto')
    callbacks =  [early_stopping_callback]
    lstm.model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks)
    train_mse, test_mse = lstm.evaluate_model(model, X_train, Y_train, X_test, Y_test, verbose)
    return train_mse, test_mse

def optimal_dropout(stock, start_date, end_date, future_gap, time_steps, split, neurons,
                    batch_size, epochs, validation_split, verbose, dropout_list):
    dropout_result = {}
    for dropout in dropout_list:    
        _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose)
        dropout_result[dropout] = testScore
    return dropout_result

def optimal_epochs(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                    neurons, batch_size, validation_split, verbose, epochs_list):
    epochs_result = {}
    for epochs in epochs_list:    
        _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose)
        epochs_result[epochs] = testScore
    return epochs_result

def optimal_neurons(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                    batch_size, epochs, validation_split, verbose, neurons_list1, neurons_list2):
    neurons_result = {}
    for lstm_neuron in neurons_list1:
        neurons = [lstm_neuron, lstm_neuron]
        for dense_neuron in neurons_list2:
            neurons.append(dense_neuron)
            neurons.append(1)
            _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                        neurons, batch_size, epochs, validation_split, verbose)
            neurons_result[str(neurons)] = testScore
            neurons = neurons[:2]
    return neurons_result

def optimal_decay(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                  neurons, batch_size, epochs, validation_split, verbose, decay_list):
    decay_result = {}
    for decay in decay_list:
        _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                    neurons, batch_size, epochs, validation_split, verbose, decay)
        decay_result[decay] = testScore
    return decay_result

def optimal_time_steps(stock, start_date, end_date, future_gap, split, dropout, neurons, batch_size,
                       epochs, validation_split, verbose, decay, time_steps_list):
    timesteps_result = {}
    for time_steps in time_steps_list:
        _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                    neurons, batch_size, epochs, validation_split, verbose, decay)
        timesteps_result[time_steps] = testScore
    return timesteps_result

def optimal_batch_size(stock, start_date, end_date, future_gap, time_steps, split, dropout, neurons,
                       epochs, validation_split, verbose, decay, batch_size_list):
    batch_size_result = {}
    for batch_size in batch_size_list:
        _, testScore = evaluate_lstm(stock, start_date, end_date, future_gap, time_steps, split, dropout,
                                    neurons, batch_size, epochs, validation_split, verbose, decay)
        batch_size_result[batch_size] = testScore
    return batch_size_result