from machine_learning.development.optimized_lstm import lstm
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

#future_gap test
future_gap_list = [1, 5, 20]
future_gap_dic = {
    1  : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    5  : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    20 : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
}
future_gap_plots = {
    1  : None,
    5  : None,
    20 : None
}

for future_gap in future_gap_list:
    start_date = dates_dic[stock][0]
    end_date = dates_dic[stock][1]
    normalized_metrics, inv_normalized_metrics, df = lstm.final_test_lstm(stock, start_date, 
    end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, validation_split, 
    verbose, callbacks)
    future_gap_dic[future_gap][0] = normalized_metrics
    future_gap_dic[future_gap][1] = inv_normalized_metrics
    future_gap_plots[future_gap] = df

print(future_gap_dic)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

df = future_gap_plots[1]
ax1.plot(df.index, df["Actual"], label='Actual')
ax1.plot(df.index, df["Prediction"], label='Prediction')
ax1.set_title('Future Gap = 1')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend(loc="best")
ax1.grid(True)

df = future_gap_plots[5]
ax2.plot(df.index, df["Actual"], label='Actual')
ax2.plot(df.index, df["Prediction"], label='Prediction')
ax2.set_title('Future Gap = 5')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend(loc="best")
ax2.grid(True)

df = future_gap_plots[20]
ax3.plot(df.index, df["Actual"], label='Actual')
ax3.plot(df.index, df["Prediction"], label='Prediction')
ax3.set_title('Future Gap = 20')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend(loc="best")
ax3.grid(True)

fig.tight_layout()
plt.show()