## Algorithms Evaluation

|  Variable  | Value       | Description |
| :--------- | :---------- | :---------- |
| stock      | ^GSPC       | S&P 500 |
| start_date | 1950-01-01  | stock historical data start date |
| end_date   | 2017-12-31  | stock historical data end date |
| window     | 1           | window for computing rolling statistics |
| future_gap | 1           | how far (trading days) into the future is the prediction |
| split      | 0.8         | training-testing dataset split |

### Evaluation metrics
    1. Loss     : RMSE of the normalized dataset, values range between [0, 1]
    2. Accuracy : R^2 (Coefficient of determination)

* <strong> Optimized LSTM </strong>
```sh
python -m machine_learning.optimized_lstm.lstm_main
```
| Future Gap | Loss (RMSE) | Accuracy (R^2) |
| :--------: | :---------: | :------------: |
| 1 day      | x           | y              |
| 1 week     | x           | y              |
| 1 month    | x           | y              |

![Optimized LSTM](https://github.com/ahmedhamdi96/ML4T/blob/master/results/optimized_lstm.png)

* <strong> Optimized FFNN </strong>
```sh
python -m machine_learning.optimized_ffnn.ffnn_main
```
| Future Gap | Loss (RMSE) | Accuracy (R^2) |
| :--------: | :---------: | :------------: |
| 1 day      | x           | y              |
| 1 week     | x           | y              |
| 1 month    | x           | y              |

![Optimized FFNN](https://github.com/ahmedhamdi96/ML4T/blob/master/results/optimized_ffnn.png)

* <strong> New kNN </strong>
```sh
python -m machine_learning.new_regression.new_kNN_regression
```
| Future Gap | Loss (RMSE) | Accuracy (R^2) |
| :--------: | :---------: | :------------: |
| 1 day      | x           | y              |
| 1 week     | x           | y              |
| 1 month    | x           | y              |

![New kNN](https://github.com/ahmedhamdi96/ML4T/blob/master/results/new_knn.png)

* <strong> New Linear Regression </strong>
```sh
python -m machine_learning.new_regression.new_linear_regression
```
| Future Gap | Loss (RMSE) | Accuracy (R^2) |
| :--------: | :---------: | :------------: |
| 1 day      | x           | y              |
| 1 week     | x           | y              |
| 1 month    | x           | y              |

![New Linear Regression](https://github.com/ahmedhamdi96/ML4T/blob/master/results/new_lin_reg.png)

## Hyperparameter Tuning

* <strong> LSTM </strong>
```sh
python -m machine_learning.optimized_lstm.hyperparam_tune_main
```
*Time Elapsed: 26.5 hours*
| Hyperparameter | Optimal Value |
| :------------: | :-----------: |
| Dropout        | 0.3           |
| Neurons        | [256, 256, 16, 1] |
| Decay          | 0.1           |
| Time Steps     | 5             |
| Batch Size     | 4096          |
| Epochs         | 200           |

![LSTM Hyperparam Tune 1](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_lstm1.png)
![LSTM Hyperparam Tune 2](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_lstm2.png)

* <strong> FFNN </strong>
```sh
python -m machine_learning.optimized_ffnn.ffnn_hyperparam_tune_main
```
*Time Elapsed: x hours*
| Hyperparameter | Optimal Value |
| :------------: | :-----------: |
| Dropout        | x             |
| Neurons        | x             |
| Decay          | x             |
| Batch Size     | x             |
| Epochs         | x             |

![FFNN Hyperparam Tune 1](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_ffnn1.png)
![FFNN Hyperparam Tune 2](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_ffnn2.png)