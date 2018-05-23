from utils.util import plot_data
from machine_learning.final.models import lstm
from machine_learning.final.models import ffnn
from machine_learning.final.models import lin_reg
from machine_learning.final.models import knn_reg
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

stock = 'AAPL'
dates_dic = {
    'AAPL'  : ['2017-01-01', '2018-01-01']
}
metrics_dic = {
    'LSTM'   : [],
    'FFNN'   : [],
    'LinReg' : [],
    'kNNReg' : []
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
end_date, window, future_gap, time_steps, neurons, drop_out, batch_size, epochs, validation_split, 
verbose, callbacks)
metrics_dic['LSTM'] = normalized_metrics
plot_data(df, stock+" 2017 Price Forecast (LSTM)", "Date", "Price", show_plot=False)

#FFNN
neurons = [256, 256, 64, 1]
batch_size = 128
epochs = 200

normalized_metrics, inv_normalized_metrics, df = ffnn.final_test_ffnn(stock, start_date, 
end_date, window, future_gap, neurons, drop_out, batch_size, epochs, validation_split, 
verbose, callbacks)
metrics_dic['FFNN'] = normalized_metrics
plot_data(df, stock+" 2017 Price Forecast (FFNN)", "Date", "Price", show_plot=False)

#LinReg
normalized_metrics, inv_normalized_metrics, df = lin_reg.final_test_linreg(stock, start_date, 
end_date, window, future_gap)
metrics_dic['LinReg'] = normalized_metrics
plot_data(df, stock+" 2017 Price Forecast (LinReg)", "Date", "Price", show_plot=False)


print(metrics_dic)
plt.show()