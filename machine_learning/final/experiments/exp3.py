from utils.util import plot_data
from machine_learning.final.models import lstm
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#companies
stocks_list = ['FB', 'AAPL', 'TSLA', 'AMZN']
dates_dic = {
    'FB'  : ['2017-12-01', '2018-05-01'],
    'AAPL': ['2012-08-01', '2013-08-01'],
    'TSLA': ['2013-08-01', '2014-01-01'],
    'AMZN': ['2017-08-01', '2018-04-01'],
}
metrics_dic = {
    'FB'   : [],
    'AAPL' : [],
    'TSLA' : [],
    'AMZN' : []
}

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

for stock in stocks_list:
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    normalized_metrics,_, df = lstm.final_test_lstm(stock, start_date, end_date, window, future_gap, time_steps,
    neurons, drop_out, batch_size, epochs, validation_split, verbose, callbacks)
    metrics_dic[stock] = normalized_metrics
    plot_data(df, stock+" Price Forecast", "Date", "Price", show_plot=False)

print(metrics_dic)
plt.show()