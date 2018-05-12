## Algorithms Evaluation

|  Variable  | Value       | Description |
| :--------- | :---------- | :---------- |
| stock      | ^GSPC       | S&P 500 |
| start date | 1950-01-01  | stock historical data start date |
| end date   | 2017-12-31  | stock historical data end date |
| window     | 5           | window for computing rolling statistics |
| future gap | 5           | how far (trading days) into the future is the prediction |
| split      | 0.8         | training-testing dataset split |

### Evaluation metrics
*metrics are applied on the normalized dataset, where the values are in the range [0, 1]*

    1. Loss
        *RMSE : accumulation of all errors, RMSE value represents dollar value
        *MAPE : accumulation of all error percentages, MAPE value represents percentage value

    2. Accuracy
        *Correlation : linear relationship between predictions and actual values, range: [-1, 1]
        *r-squared   : how close predictions are to actual prices, range: [0, 1]

* <strong> Optimized LSTM </strong>
```sh
python -m machine_learning.optimized_lstm.lstm_main
```
| Future Gap | RMSE | MAPE | Corr | R^2 |
| :--------: | :--: | :--: | :--: | :-: |
| 1 day      | x    | y    | i    | j   |
| 1 week     | 0.056| 5.075| 0.092| 0.788|
| 1 month    | x    | y    | i    | j   |

![Optimized LSTM](https://github.com/ahmedhamdi96/ML4T/blob/master/results/optimized_lstm.png)

* <strong> Optimized FFNN </strong>
```sh
python -m machine_learning.optimized_ffnn.ffnn_main
```
| Future Gap | RMSE | MAPE | Corr | R^2 |
| :--------: | :--: | :--: | :--: | :-: |
| 1 day      | x    | y    | i    | j   |
| 1 week     | 0.013| 1.786| 0.998| 0.994|
| 1 month    | x    | y    | i    | j   |

![Optimized FFNN](https://github.com/ahmedhamdi96/ML4T/blob/master/results/optimized_ffnn.png)

* <strong> New Linear Regression </strong>
```sh
python -m machine_learning.new_regression.new_linear_regression
```
| Future Gap | RMSE | MAPE | Corr | R^2 |
| :--------: | :--: | :--: | :--: | :-: |
| 1 day      | x    | y    | i    | j   |
| 1 week     | 0.011| 1.597| 0.998| 0.995|
| 1 month    | x    | y    | i    | j   |

![New Linear Regression](https://github.com/ahmedhamdi96/ML4T/blob/master/results/new_lin_reg.png)

## Hyperparameter Tuning

* <strong> LSTM </strong>
```sh
python -m machine_learning.optimized_lstm.hyperparam_tune_main
```
*Time Elapsed: 25 hours*

| Hyperparameter | Optimal Value |
| :------------: | :-----------: |
| Dropout        | 0.2           |
| Neurons        | [256, 256, 32, 1] |
| Decay          | 0.1           |
| Time Steps     | 5             |
| Batch Size     | 2048          |
| Epochs         | 300           |

![LSTM Hyperparam Tune 1](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_lstm1.png)
![LSTM Hyperparam Tune 2](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_lstm2.png)

* <strong> FFNN </strong>
```sh
python -m machine_learning.optimized_ffnn.ffnn_hyperparam_tune_main
```
*Time Elapsed: 5.3 minutes*

| Hyperparameter | Optimal Value |
| :------------: | :-----------: |
| Dropout        | 0.8           |
| Neurons        | [256, 256, 64, 1] |
| Decay          | 0.1           |
| Batch Size     | 128           |
| Epochs         | 200           |

![FFNN Hyperparam Tune 1](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_ffnn1.png)
![FFNN Hyperparam Tune 2](https://github.com/ahmedhamdi96/ML4T/blob/master/results/hyperparam_tune_ffnn2.png)