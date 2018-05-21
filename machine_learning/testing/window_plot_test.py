from machine_learning.optimized_lstm import lstm
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

stock = 'AAPL'
dates_dic = {
    'AAPL'  : ['2017-01-01', '2018-01-01']
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

#window test
window_list = [2,3,4,5]
window_dic = {
    2 : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None],
    3 : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None],
    4 : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None],
    5 : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None]
}

window = 5

for window in window_list:
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    normalized_metrics, inv_normalized_metrics, df = lstm.final_test_lstm(stock, start_date, 
    end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, validation_split, 
    verbose, callbacks)
    window_dic[window][0] = normalized_metrics
    window_dic[window][1] = inv_normalized_metrics
    window_dic[window][2] = df

fig1, (ax1, ax2) = plt.subplots(2, 1)
fig2, (ax3, ax4) = plt.subplots(2, 1)

df = window_dic[2][2]
ax1.plot(df.index, df["Actual"], label='Actual')
ax1.plot(df.index, df["Prediction"], label='Prediction')
ax1.set_title('Window = 2')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend(loc="best")
ax1.grid(True)

df = window_dic[3][2]
ax2.plot(df.index, df["Actual"], label='Actual')
ax2.plot(df.index, df["Prediction"], label='Prediction')
ax2.set_title('Window = 3')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend(loc="best")
ax2.grid(True)

df = window_dic[4][2]
ax3.plot(df.index, df["Actual"], label='Actual')
ax3.plot(df.index, df["Prediction"], label='Prediction')
ax3.set_title('Window = 4')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend(loc="best")
ax3.grid(True)

df = window_dic[5][2]
ax4.plot(df.index, df["Actual"], label='Actual')
ax4.plot(df.index, df["Prediction"], label='Prediction')
ax4.set_title('Window = 5')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price')
ax4.legend(loc="best")
ax4.grid(True)

fig1.tight_layout()
fig2.tight_layout()
plt.show()