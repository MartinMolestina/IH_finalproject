
#FUNCTIONS
from StockFunctions import yahoo_get, plot_corr, feature_eng_yahoo, plot_stock
from StockFunctions import LR_train, LR_predict, plot_predictions, twitter, get_tweets_hash
from StockFunctions import summary
from montecarlo import mcmc

import datetime



if __name__ == "__main__":

    #Ask user for comapnies ticker
    ticker = str(input("Company's Stock Ticker: "))
    days_predict = int(input('Prediction range (days): '))
    mcmc_sim = int(input('MCMC simulation (days): '))


    #Request YahooFinance library for companies historic data
    raw_data = yahoo_get(ticker, '2015-01-01', str(datetime.datetime.now().date()))
    plot_corr(raw_data, 10, 'Raw Data Correlation {}'.format(ticker))
    clean_data = feature_eng_yahoo(raw_data, days_predict)
    plot_stock(clean_data, 'Adj_Close', 15 ,'{} Stock Price'.format(ticker))
    plot_corr(clean_data, 10,'Clean Data Correlation {}'.format(ticker))
    
    #Train model and predict Close Stock Prices
    score, reg = LR_train(clean_data, days_predict)
    predictions, predicted_df = LR_predict(clean_data, reg, days_predict)
    plot_predictions(predicted_df,['future_price{}d'.format(days_predict), 'Predictions'] , 10, '{} Stock Predictions'.format(ticker))
    
    #Retrieve related tweets
    api = twitter()
    tweets = get_tweets_hash('#{}'.format(ticker), api)
    
    #Create summary dataframe
    stock_summary = summary(clean_data, tweets, predictions, days_predict)

    #Markov Chain Monte Carlo
    probability_table = mcmc(raw_data, mcmc_sim, ticker)

    #Print desired values
    print(stock_summary)


