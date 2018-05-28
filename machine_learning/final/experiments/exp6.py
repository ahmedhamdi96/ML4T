from utils.util import plot_data
from machine_learning.final.models import lstm
from machine_learning.final.models import ffnn
from machine_learning.final.models import lstm_ffnn
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#models comparison
stock = 'AAPL'
dates_dic = {
    'AAPL'  : ['2017-01-01', '2018-01-01']
}
time_steps_list = [1, 2, 3, 4, 5]

window = 2
future_gap = 1
time_steps = 1
neurons = [256, 256, 32, 1]
drop_out = 0.2                                   
batch_size = 2048
epochs = 300
validation_split = 0.1
verbose = 1
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                        patience=50, verbose=verbose, mode='auto')
callbacks = [early_stopping_callback] 
start_date = dates_dic[stock][0]
end_date = dates_dic[stock][1]

lstm_time_steps_metrics = {
    1 : [],
    2 : [],
    3 : [],
    4 : [],
    5 : []
}

#LSTM
for time_steps in time_steps_list:
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    normalized_metrics, _, df = lstm.final_test_lstm(stock, start_date, 
    end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, validation_split, 
    verbose, callbacks)
    lstm_time_steps_metrics[time_steps] = normalized_metrics
    plot_data(df, 'LSTM Time Steps = '+str(time_steps), "Date", "Price", show_plot=False)

#LSTM_FFNN
neurons = [256, 256, 64, 1]
batch_size = 128
epochs = 200

lstm_ffnn_time_steps_metrics = {
    1 : [],
    2 : [],
    3 : [],
    4 : [],
    5 : []
}

for time_steps in time_steps_list:
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    normalized_metrics, _, df = lstm_ffnn.final_test_lstm_ffnn(stock, start_date, 
    end_date, window, time_steps, future_gap, neurons, drop_out, batch_size, epochs, validation_split, 
    verbose, callbacks)
    lstm_ffnn_time_steps_metrics[time_steps] = normalized_metrics
    plot_data(df, 'LSTM FNNN Time Steps = '+str(time_steps), "Date", "Price", show_plot=False)

print(lstm_time_steps_metrics)
print(lstm_ffnn_time_steps_metrics)
plt.show()