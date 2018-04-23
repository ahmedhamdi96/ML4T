from machine_learning.optimized_ffnn import ffnn
from keras.callbacks import EarlyStopping

def evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                  neurons, batch_size, epochs, validation_split, verbose, decay=0.0):

    dataframe, _ = ffnn.bulid_TIs_dataset(stock, start_date, end_date, window)
    dataset = dataframe.values
    X_train, Y_train, X_test, Y_test = ffnn.ffnn_dataset_reshape(dataset, future_gap, split)
    features = X_train.shape[1]
    model = ffnn.build_model(features, neurons, dropout, decay)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                            patience=50, verbose=verbose, mode='auto')
    callbacks =  [early_stopping_callback]
    ffnn.model_fit(model, X_train, Y_train, batch_size, epochs, validation_split, verbose, callbacks)
    train_mse, test_mse = ffnn.evaluate_model(model, X_train, Y_train, X_test, Y_test, verbose)
    return train_mse, test_mse

def optimal_dropout(stock, start_date, end_date, window, future_gap, split, neurons,
                    batch_size, epochs, validation_split, verbose, dropout_list):
    dropout_result = {}
    for dropout in dropout_list:
        print("\n> testing droput: (%.1f)..." %(dropout))    
        _, testScore = evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose)
        dropout_result[dropout] = testScore
    return dropout_result

def optimal_epochs(stock, start_date, end_date, window, future_gap, split, dropout,
                    neurons, batch_size, validation_split, verbose, epochs_list):
    epochs_result = {}
    for epochs in epochs_list: 
        print("\n> testing epochs: (%d)..." %(epochs))    
        _, testScore = evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose)
        epochs_result[epochs] = testScore
    return epochs_result

def optimal_neurons(stock, start_date, end_date, window, future_gap, split, dropout,
                    batch_size, epochs, validation_split, verbose, neurons_list1, neurons_list2):
    neurons_result = {}
    for ffnn_neuron in neurons_list1:
        neurons = [ffnn_neuron, ffnn_neuron]
        for dense_neuron in neurons_list2:
            neurons.append(dense_neuron)
            neurons.append(1)
            print("\n> testing neurons: (%s)..." %(str(neurons))) 
            _, testScore = evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                                         neurons, batch_size, epochs, validation_split, verbose)
            neurons_result[str(neurons)] = testScore
            neurons = neurons[:2]
    return neurons_result

def optimal_decay(stock, start_date, end_date, window, future_gap, split, dropout,
                  neurons, batch_size, epochs, validation_split, verbose, decay_list):
    decay_result = {}
    for decay in decay_list:
        print("\n> testing decay: (%.1f)..." %(decay))
        _, testScore = evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose, decay)
        decay_result[decay] = testScore
    return decay_result

def optimal_batch_size(stock, start_date, end_date, window, future_gap, split, dropout, neurons,
                       epochs, validation_split, verbose, decay, batch_size_list):
    batch_size_result = {}
    for batch_size in batch_size_list:
        print("\n> testing batch size: (%d)..." %(batch_size))
        _, testScore = evaluate_ffnn(stock, start_date, end_date, window, future_gap, split, dropout,
                                     neurons, batch_size, epochs, validation_split, verbose, decay)
        batch_size_result[batch_size] = testScore
    return batch_size_result