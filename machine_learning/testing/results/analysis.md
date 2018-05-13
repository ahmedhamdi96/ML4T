## Analysis with PAL

### Sudden Changes vs Normal Movements

This forecast is used to predict the tesla stock for a duration between 01/01/2013 and 01/06/2013, PAL is also used to analyze the behaviour of the model during two different periods a stock usually goes through; a normal movement, where the stock price fluctuates  with no dramatic change, the other period is a sudden change period, where the stock moves violently either upwards, downwards, or up and down with high volatility.

![SvN](https://github.com/ahmedhamdi96/ML4T/blob/master/results/sudden_vs_normal.png)

> Up until 07/05/2013, the stock price movement exhibits  a normal movement, no violent trajectories appear. 
> This is when the model performs best. The forecast does not lag the actual price, and follows the same trend 
> and movement of the actual price. Starting from 07/05/2013, the stock moves up with a steep trajectory, and 
> during that sudden change is when the model performs poorly. Upon researching news about Tesla on May/2013, 
> it was discovered that the company reported its first quarterly profit and its flagship at that time, the 
> Model S, received the best review of any car in Consumer Reports magazine's history, see the report 
> [here](http://money.cnn.com/2013/05/10/investing/tesla-stock). These postive news caused an unexpected and 
> sudden surge in Tesla's stock price. A hypothesis that can be proposed from that is that the model is capable 
> of predicting the price and the fluctuations in price caused by the stock market movement, but when external 
> events that impact the stock price suddenly, the model naturally does not pick up on these events.

![Lag](https://github.com/ahmedhamdi96/ML4T/blob/master/results/sudden_vs_normal_lag.png)

> This plot shows the frequency of when was the prediction closest to the actual price, the day lag indicates 
> the number of days it took for the forecast to best match the actual price. 

![Daily Lag](https://github.com/ahmedhamdi96/ML4T/blob/master/results/sudden_vs_normal_daily_lag.png)

> This plot follows the same timeline of the forecast on the x-axis, against the lag on the y-axis. This plot 
> supports the hypothesis, mentioned earlier, the model finds the closest prediction to the actual price early 
> on during the normal movement phase, and lags at the end of the timeline during the sudden change phase.

### Stable (Amazon) Stock vs Volatile(Tesla) Stock Forecast

![Stable](https://github.com/ahmedhamdi96/ML4T/blob/master/results/stable.png)

![Stable Lag](https://github.com/ahmedhamdi96/ML4T/blob/master/results/stable_lag.png)

![Volatile](https://github.com/ahmedhamdi96/ML4T/blob/master/results/volatile.png)

![Volatile Lag](https://github.com/ahmedhamdi96/ML4T/blob/master/results/volatile_lag.png)