from utils.util import get_data, plot_data
import machine_learning.knn as knn
import pandas as pd

def compute_momentum_ratio(prices, window):
    momentum = (prices/prices.shift(periods = -window)) - 1
    return momentum

def compute_sma_ratio(prices, window):
    sma = (prices / prices.rolling(window = window).mean()) - 1
    return sma

def compute_bollinger_bands_ratio(prices, window):
    bb = prices - prices.rolling(window = window).mean()
    bb = bb / (2 * prices.rolling(window = window).std())
    #bb = bb - 1
    return bb

def normalize_ratio(prices):
    return prices - prices.mean()/prices.std()

def evaluate_predictions(learner, input_values, actual_values, trading_days_df):
    predicted_values = learner.query(input_values)

    trading_days_df['predicted_values'] = predicted_values
    trading_days_df['actual_values'] = actual_values

    rmse = (((actual_values - predicted_values) ** 2).sum() / actual_values.shape[0]) ** 0.5
    correlation = trading_days_df.ix[:, ['predicted_values', 'actual_values']].corr()

    print("RMSE:", rmse)
    print("Correlation:", correlation.ix[0, 1])

    df = trading_days_df.ix[:, ['predicted_values', 'actual_values']]
    plot_data(df, "Predicted vs Actual Prices", "Date", "Price", leg_loc="best")

def main():
    training_start_date = '01/01/2015'
    training_end_date = '31/12/2016'

    testing_start_date = '01/01/2017'
    testing_end_date = '31/12/2017'

    stock = 'IBM'

    training_prices_df = get_data([stock], training_start_date, training_end_date)
    testing_prices_df = get_data([stock], testing_start_date, testing_end_date)

    #learner input, a.k.a. Xs, a.k.a. features
    future_gap = 5 #1 trading week

    #Training Phase
    training_date_range = pd.date_range(training_start_date, training_end_date)
    training_df = pd.DataFrame(index=training_date_range)

    training_df['actual_prices'] = training_prices_df[stock]
    training_df['bolinger_band'] = compute_bollinger_bands_ratio(training_prices_df[stock], future_gap)
    training_df['momentum'] = compute_momentum_ratio(training_prices_df[stock], future_gap)
    training_df['volatility'] = ((training_prices_df[stock]/training_prices_df[stock].shift(periods= -1)) - 1).rolling(window=future_gap).std()
    training_df['y_values'] = training_prices_df[stock].shift(periods = -future_gap)
    training_df = training_df.dropna(subset=['actual_prices'])

    trainX = training_df.iloc[future_gap-1:, :-1]
    trainY = training_df.iloc[future_gap-1:, -1]

    #Testing Phase
    testing_date_range = pd.date_range(testing_start_date, testing_end_date)
    testing_df = pd.DataFrame(index=testing_date_range)

    testing_df['actual_prices'] = testing_prices_df[stock]
    testing_df['bolinger_band'] = compute_bollinger_bands_ratio(testing_prices_df[stock], future_gap)
    testing_df['momentum'] = compute_momentum_ratio(testing_prices_df[stock], future_gap)
    testing_df['volatility'] = ((testing_prices_df[stock]/testing_prices_df[stock].shift(periods= -1)) - 1).rolling(window=future_gap).std()
    testing_df['y_values'] = testing_prices_df[stock].shift(periods = -future_gap)
    testing_df = testing_df.dropna(subset=['actual_prices'])

    testX = testing_df.iloc[:, 0:-1]
    testY = testing_df.iloc[:, -1]

    #kNN Learner
    knn_learner = knn.knn(3)
    knn_learner.train(trainX, trainY)

    #trading days df
    training_days_df = pd.DataFrame(index=training_date_range)
    training_days_df['actual_prices'] = training_prices_df[stock]
    training_days_df = training_days_df.dropna(subset=['actual_prices'])

    testing_days_df = pd.DataFrame(index=testing_date_range)
    testing_days_df['actual_prices'] = testing_prices_df[stock]
    testing_days_df = testing_days_df.dropna(subset=['actual_prices'])

    #Insample Testing
    print("Insample Testing")
    evaluate_predictions(knn_learner, trainX, trainY, training_days_df)
    #Outsample Testing
    print("\nOutsample Testing")
    evaluate_predictions(knn_learner, testX, testY, testing_days_df)

if __name__ == "__main__":
    main()