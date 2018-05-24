from machine_learning.final.models import lstm
from machine_learning.final.evaluation.metrics import compute_lag_metric
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#sudden vs normal
stock = 'TSLA'
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
start_date = dates_dic[stock][0]
end_date = dates_dic[stock][1]
#LSTM
normalized_metrics, inv_normalized_metrics, df = lstm.final_test_lstm(stock, start_date, 
end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, 
validation_split, verbose, callbacks)
#PAL
lookup = 5
lag_list = compute_lag_metric(df['Actual'], df['Prediction'], lookup, stock)
#Price Forecast Plot
df_test = df[:len(df)-lookup+1]
ax = df.plot(title=stock+" Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend(loc="best")
ax.grid(True)
ax.annotate('Normal Movement', xy=('2013-02-15', 40), xytext=('2013-03-05', 50), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
ax.annotate('Sudden Change', xy=('2013-05-10', 55), xytext=('2013-03-05', 70), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
#Price Forecast and PAL Overlay Plot
ax = df_test.plot(title=stock+" Price Forecast and PAL Overlay")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend(loc="best")
ax.grid(True)
ax.annotate('Normal Movement', xy=('2013-02-15', 40), xytext=('2013-03-05', 50), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
ax.annotate('Sudden Change', xy=('2013-05-10', 55), xytext=('2013-03-05', 70), fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.1, headwidth=8))
ax1 = ax.twinx()
ax1.scatter(df_test.index, lag_list, c='g')
ax1.set_ylabel("PAL")

plt.show()