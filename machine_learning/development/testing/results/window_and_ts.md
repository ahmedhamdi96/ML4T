## Window and Time Steps Test

### Time Steps Test

| Time Steps | RMSE | MAPE | Corr | R^2  |
| :--------: | :--: | :--: | :--: | :--: |
| 1 | 0.0317 | 5.26 | 0.993 | 0.982 |
| 2 | 0.0338 | 5.45 | 0.990 | 0.979 |
| 3 | 0.0452 | 7.89 | 0.988 | 0.961 |
| 4 | 0.0462 | 7.77 | 0.985 | 0.959 |
| 5 | 0.0538 | 8.91 | 0.982 | 0.942 |

Winner: 1

### Window Test

| Window | RMSE | MAPE | Corr | R^2  |
| :----: | :--: | :--: | :--: | :--: |
| 2 | 0.0299 | 4.84 | 0.994 | 0.984 |
| 3 | 0.0294 | 5.31 | 0.993 | 0.985 |
| 4 | 0.0336 | 7.85 | 0.992 | 0.981 |
| 5 | 0.0287 | inf  | 0.993 | 0.986 |

Winner: The metrics are not decisive enough, so a plot test could help.

![Window 2,3](https://github.com/ahmedhamdi96/ML4T/blob/master/results/window_test_1.png)
![Window 4,5](https://github.com/ahmedhamdi96/ML4T/blob/master/results/window_test_2.png)

