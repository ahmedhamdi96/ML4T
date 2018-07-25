# ML4T
*Machine Learning for Trading*

## Project Overview
*GUC 2018 Bachelor Thesis Project*

Stock market prediction is an interesting realm to test the capabilities of machine learning
on. The nature of the stock market is volatile, sophisticated, and very sensitive to external
information, which makes it difficult to predict. Different machine learning models
are developed to forecast future stock prices. Using historical stock market data, technical
indicators are computed and used along with a stock’s price as features associated
with a target output, which is the future stock price. This provides a dataset that the
machine learning models use to train upon, and thus the models become capable of predicting
future prices. The models used are: linear regressor, kNN regressor, Feedforward
Neural Network (FFNN), and Long Short Term Memory (LSTM) Recurrent Neural Network
(RNN). The prediction models are compared and evaluated using different metrics.
Several case studies are performed to evaluate the performance of the machine learning
models. From the case studies, few insights have been made:

1. The LSTM RNN outperformed all the other models.
2. The LSTM RNN model is capable of accurately predicting the next-day price unless a major external event impacts the stock price suddenly.
3. The LSTM RNN model naturally lags on picking up on external events that impact the stock price suddenly.

## Algorithms Evaluation
*Development Phase*

* [Original](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/original_evaluation.md)
* [New](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/new_evaluation.md)

## Testing
*Testing Phase*

    * Considering that the LSTM model is regarded as the flagship machine learning model in this project, 
    it is the one used in this testing section.

    * The model is trained on the period starting from a company's first public trading day till the day 
    before the required testing period.

### Companies During Times of Change
*Predicting Stock prices for a portfolio of 4 companies during different interesting time periods*

* **[Facebook](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/facebook.md)**

  *Facebook started trading publicly on 18/05/2012.*

  * Facebook–Cambridge Analytica data scandal, [January/2018 - March/2018]

    Amid the scandal and Mark Zuckerburg's public hearing, Facebook's stock price fell.

* **[Apple](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/apple.md)**

  *Apple started trading publicly on 12/12/1980.*

  * Apple's first free fall, [September/2012 - June/2013]

    Apple faced multiple hardships during this period; earnings were no longer growing, 
    low-priced phones were capturing most of the smartphone market share over the iPhone,
    and the company entered the "post-Steve Jobs" era where the company's next generation 
    of leaders and products were in question.

* **[Tesla](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/tesla.md)**

  *Tesla started trading publicly on 29/06/2010.*

  * Disappointing Q3 Reports, [September/2013 - November/2013]

    Tesla reported disappointing third quarter financial results. In addition, 
    a third widely-reported fire involving a Model S in just two months was 
    putting Tesla under heat.

* **[Amazon](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/amazon.md)**

  *Amazon started trading publicly on 15/05/1997.*

  * Exceeding Q3 expectations, [September/2017 - February/2018]

    Amazon's Q3 reports showed an increase in profits, an acceleration in revenue growth, an increase 
    in AWS' operating income, and the success of Alexa-enabled devices.

### Window and Time Steps Test
A test to determine the optimal window and time steps. See results [here](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/window_and_ts.md).

### Evaluation Metrics
New metrics to evaluate the performance of the model over different future gaps. See results [here](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/eval.md).

### Future Gap Test
A test to compare between the linear regressor, FFNN, and LSTM RNN over different future gaps. See results [here](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/future_gap.md).

## Analysis
*Analysing the tests using a novel metric*

To analyse the forecast and evaluate how fast does the model predict the closest price to the actual, a lag metric is created.
The **_Prediction-Actual Lag (PAL)_** metric works as follows: 
The future gap chosen when making the forecast indicates how far into the future should a prediction be, for example if the future gap is set to 1, the forecast is a next-trading-day forecast. The actual prices are traversed and compared with the predictions, each actual price datapoint is compared against a number of the prediction data points, that number is the future gap, so if the future gap is set to 5, then each actual datapoint is compared to the corresponding prediction datapoint and the 4 next to it. See **_PAL_** in action [here](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/development/testing/results/analysis.md).

## Software and Libraries
*This project uses the following software and Python libraries:*

* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/index.html)
* [SciPy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org)
* [Keras](https://keras.io/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/doc_index.html)
