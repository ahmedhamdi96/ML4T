from machine_learning.optimized_lstm import lstm
from keras.callbacks import EarlyStopping

stocks_list = ['FB', 'AAPL', 'TSLA', 'AMZN']
show_plot = len(stocks_list)
dates_dic = {
    'FB'  : ['2017-12-01', '2018-05-01'],
    'AAPL': ['2012-08-01', '2013-08-01'],
    'TSLA': ['2013-08-01', '2014-01-01'],
    'AMZN': ['2017-08-01', '2018-04-01'],
    }

window = 3
future_gap = 1
time_steps = 5
neurons = [256, 256, 32, 1]
drop_out = 0.2                                   
batch_size = 2048
epochs = 300
validation_split = 0.1
verbose = 1
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, 
                                        patience=50, verbose=verbose, mode='auto')
callbacks = [early_stopping_callback] 

for stock in stocks_list:
    show_plot -= 1
    show_plot_flg = True if show_plot == 0  else False
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    lstm.test_lstm(stock, start_date, end_date, window, future_gap, time_steps,
                neurons, drop_out, batch_size, epochs, validation_split, verbose, callbacks, show_plot_flg)