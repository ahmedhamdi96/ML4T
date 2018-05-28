## Future Gap Test

### Microsoft 2017 Stock Price Forecast

#### LSTM
| Future Gap | RMSE | NRMSE | MAE | Corr | R^2  |
| :--------: | :--: | :--: | :--: | :--: | :--: |
| 1 Day  | 0.0273 | 0.0676 | 0.0184 | 0.995 | 0.991 |
| 2 Days | 0.0369 | 0.0909 | 0.0254 | 0.992 | 0.983 |
| 3 Days | 0.0437 | 0.1070 | 0.0314 | 0.989 | 0.976 | 
| 4 Days | 0.0496 | 0.1210 | 0.0363 | 0.985 | 0.969 |
| 5 Days | 0.0568 | 0.1380 | 0.0421 | 0.981 | 0.959 |

#### Linear Regressor
| Future Gap | RMSE | NRMSE | MAE | Corr | R^2  |
| :--------: | :--: | :--: | :--: | :--: | :--: |
| 1 Day  | 0.0275 | 0.0679 | 0.0185 | 0.993 | 0.990 |
| 2 Days | 0.0372 | 0.0917 | 0.0260 | 0.992 | 0.983 |
| 3 Days | 0.0441 | 0.1080 | 0.0317 | 0.989 | 0.976 | 
| 4 Days | 0.0504 | 0.1230 | 0.0366 | 0.985 | 0.968 |
| 5 Days | 0.0572 | 0.1390 | 0.0422 | 0.981 | 0.958 |

#### FFNN
| Future Gap | RMSE | NRMSE | MAE | Corr | R^2  |
| :--------: | :--: | :--: | :--: | :--: | :--: |
| 1 Day  | 0.0376 | 0.0931 | 0.0278 | 0.994 | 0.982 |
| 2 Days | 0.0474 | 0.1170 | 0.0335 | 0.991 | 0.972 |
| 3 Days | 0.0691 | 0.1700 | 0.0501 | 0.984 | 0.939 | 
| 4 Days | 0.0535 | 0.1310 | 0.0389 | 0.982 | 0.964 |
| 5 Days | 0.0709 | 0.1729 | 0.0512 | 0.972 | 0.936 |

*Shown below are the forecasts of the LSTM RNN model*

![Future Gap](https://github.com/ahmedhamdi96/ML4T/blob/master/results/experiments/exp4/gap1.png)

![Future Gap](https://github.com/ahmedhamdi96/ML4T/blob/master/results/experiments/exp4/gap2.png)

![Future Gap](https://github.com/ahmedhamdi96/ML4T/blob/master/results/experiments/exp4/gap3.png)

![Future Gap](https://github.com/ahmedhamdi96/ML4T/blob/master/results/experiments/exp4/gap4.png)

![Future Gap](https://github.com/ahmedhamdi96/ML4T/blob/master/results/experiments/exp4/gap5.png)