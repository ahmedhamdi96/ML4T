# ML4T
*Machine Learning for Trading*

## Project Overview
*GUC Bachelor Thesis Project*

This is the experimentation section of the thesis. 
* **Thesis Research Questions:**
  * *Can machine learning be used to predict future stock prices?*
  * *How does the performance of different machine learning algorithms vary?*

## Algorithms Evaluation
*Time Series Forecast Evaluation*

* [Original](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/original_evaluation.md)
* [New](https://github.com/ahmedhamdi96/ML4T/blob/master/machine_learning/new_evaluation.md)

## Testing
*Predicting Stock prices for a portfolio of 4 companies during different interesting time periods*

    * Considering that the LSTM model is regarded to be the flagship machine learning model 
    in this project, it is the one used in this testing section.

    * The model is trained on the period starting from a company's first public trading day 
    till the day before the required testing period.

* **[Facebook](https://github.com/ahmedhamdi96/ML4T/blob/master/testing/results/facebook.md)**

  *Facebook started trading publicly on 18/05/2012.*

  * Facebook–Cambridge Analytica data scandal, [January/2018 - March/2018]

    Amid the scandal and Mark Zuckerburg's public hearing, Facebook's stock price fell.

* **[Apple](https://github.com/ahmedhamdi96/ML4T/blob/master/testing/results/apple.md)**

  *Apple started trading publicly on 12/12/1980.*

  * Apple's first free fall, [September/2012 - June/2013]

    Apple faced multiple hardships during this period; earnings were no longer growing, 
    low-priced phones were capturing most of the smartphone market share over the iPhone,
    and the company entered the "post-Steve Jobs" era where the company's next generation 
    of leaders and products were in question.


* **[Tesla](https://github.com/ahmedhamdi96/ML4T/blob/master/testing/results/tesla.md)**

  *Tesla started trading publicly on 29/06/2010.*

  Tesla's stock is volatile as a result of many factors including: failing to meet plans 
  and expensive acquisitions and investments.

  * Analysts downgrades, [September/2013 - November/2013]

    Lowered volume expectations for Model X and Model 3, a lower valuation for Tesla Energy, 
    and accelerating competition in the mobility business were some of the reasons analysts
    lowered Tesla's stock price target.

* **[Amazon](https://github.com/ahmedhamdi96/ML4T/blob/master/testing/results/amazon.md)**

  *Amazon started trading publicly on 15/05/1997.*

  Overall Amazon's stock throughout its run since 1997 has been healthy, with projections mostly 
  heading upwards and with little volatility.

  * Exceeding Q3 expectations, [September/2017 - February/2018]

    Amazon's Q3 reports showed an increase in profits, an acceleration in revenue growth, an increase 
    in AWS' operating income, and the success of Alexa-enabled devices.

## Software and Libraries
This project uses the following software and Python libraries:

* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/index.html)
* [SciPy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org)
* [Keras](https://keras.io/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/doc_index.html)