#GENERAL LIBRARIES
import pandas as pd
import numpy as np
import time
import datetime
from dotenv import load_dotenv
import os

#ML LIBRARIES
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#PLOTING LIBRARIES
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

#DATA REQUESTING LIBRARIES
import quandl
from yahoofinancials import YahooFinancials
import intrinio_sdk

#TWITTER SENTIMENT ANALYSIS LIBRARIES
import tweepy as tw
from textblob import TextBlob

#FUNCTIONS
from StockFunctions import yahoo_get, plot_corr, feature_eng_yahoo, plot_stock
from StockFunctions import LR_train, LR_predict, plot_predictions, twitter, get_tweets_hash
from StockFunctions import summary




if __name__ == "__main__":

    #Ask user for comapnies ticker
    ticker = str(input("Company's Stock Ticker: "))
    days_predict = int(input('Prediction range (days): '))

    #Request YahooFinance library for companies historic data
    raw_data = yahoo_get(ticker, '2015-01-01', str(datetime.datetime.now().date()))
    plot_corr(raw_data, 10, 'Raw Data Correlation {}'.format(ticker))
    clean_data = feature_eng_yahoo(raw_data, days_predict)
    plot_stock(clean_data, 'Adj_Close', 15 ,'{} Stock Price'.format(ticker))
    plot_corr(clean_data, 10,'Clean Data Correlation {}'.format(ticker))
    
    #Train model and predict Close Stock Prices AAPL
    score, reg = LR_train(clean_data, days_predict)
    predictions, predicted_df = LR_predict(clean_data, reg, days_predict)
    plot_predictions(predicted_df,['future_price{}d'.format(days_predict), 'Predictions'] , 10, '{} Stock Predictions'.format(ticker))
    
    #Retrieve related tweets
    api = twitter()
    tweets = get_tweets_hash('#{}'.format(ticker), api)
    
    #Create summary dataframe
    stock_summary = summary(clean_data, tweets, predictions)

    #Print desired values
    print(stock_summary)


