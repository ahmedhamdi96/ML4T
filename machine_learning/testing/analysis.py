from machine_learning.optimized_lstm import lstm
from keras.callbacks import EarlyStopping

#volatile vs stable
'''stocks_list = ['TSLA', 'AMZN']
show_plot = len(stocks_list)
dates_dic = {
    'TSLA': ['2013-01-01', '2018-01-01'],
    'AMZN': ['2013-01-01', '2018-01-01'],
    }
'''
#sudden vs normal
stocks_list = ['TSLA']
show_plot = len(stocks_list)
dates_dic = {
    'TSLA': ['2013-01-01', '2013-06-01'],
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
    show_plot -= 1
    show_plot_flg = True if show_plot == 0  else False
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    lstm.test_lstm(stock, start_date, end_date, window, future_gap, time_steps,
                neurons, drop_out, batch_size, epochs, validation_split, verbose, callbacks, show_plot_flg)

#sudden vs normal forecast annotations
''' 
ax.annotate('Normal Movement', xy=('2013-02-15', 40), xytext=('2013-03-05', 50), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
ax.annotate('Sudden Change', xy=('2013-05-10', 55), xytext=('2013-03-05', 70), fontsize=10,
        arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
'''