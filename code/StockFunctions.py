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


# GETS THE API KEY FROM .ENV FILE
def get_key(d_key):
    load_dotenv('.env')
    key = os.environ.get(d_key)
    return key


# GETS DATA FROM QUANDL / DEFAULT IS BOING DATA 
def quandl_get(code="EOD/BA", start_date="1970-01-01", end_date=str(datetime.datetime.now().date())):
    return quandl.get(code, start_date=start_date, end_date=end_date)


# GETS DATA FROM YAHOOFINANCIALS
def yahoo_get(ticker='AAPL', start='1970-01-01', end=str(datetime.datetime.now().date())):
    yahoo_financials = YahooFinancials(ticker)
    dic = yahoo_financials.get_historical_price_data(start, end, 'daily')
    df = pd.DataFrame(dic[ticker]['prices'])
    df['Date'] = pd.to_datetime(df['date'], unit='s').dt.date
    df.drop(columns=['formatted_date', 'date'], inplace=True)
    return df


# CONNECTS TO TWITTER API
def twitter():
    consumer_key = get_key('twitter_api')
    consumer_secret = get_key('twitter_api_secret')
    access_token = get_key('twitter_access_token')
    access_token_secret = get_key('twitter_access_token_secret')
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api

    
# RETRIEVES ALL TWEETS WITH HASHTAG   
def get_tweets_hash(hashtag, api):
    tmp = []
    for tweet in tw.Cursor(api.search,q=hashtag ,count=100, lang="en",
                           since=str(datetime.datetime.now().date()-datetime.timedelta(days=15)),
                           tweet_mode='extended').items():
        tmp.append((tweet.created_at, tweet.full_text))
        
    df = pd.DataFrame(tmp)
    df['date'] = df[0].dt.date
    group = df.groupby(['date']).sum()
    group['Sentiment'] = group[1].apply(sentiment)
    group.columns = ['Tweets', 'Sentiment']
    return group


# KEEP SIGNIFICANT COLUMNS. ADD CHANGE AND VOLATILITY
def feature_eng_quandl(df, n=10):
    df['Volatility'] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Low']
    df['Change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open']
    df = df[['Adj_Close', 'Volatility', 'Change', 'Adj_Volume']]
    df['20ma'] = df['Adj_Close'].rolling(window=20, min_periods=0).mean()
    df.fillna('-999999', inplace=True) # N/A value treated as outlier
    #Defines forecast to predict
    forecast_column = 'Adj_Close'
    #shifts label to the past n days
    df['future_price{}d'.format(n)] = df[forecast_column].shift(-n)
    return df


# KEEP SIGNIFICANT COLUMNS. ADD CHANGE AND VOLATILITY
def feature_eng_yahoo(df, n=5):
    df.index = df['Date']
    df['Volatility'] = (df['high'] - df['low']) / df['low']
    df['Change'] = (df['close'] - df['open']) / df['open']
    df = df[['close', 'Volatility', 'Change', 'volume']]
    df.columns = ['Adj_Close', 'Volatility', 'Change', 'Adj_Volume']
    df['20ma'] = df['Adj_Close'].rolling(window=20, min_periods=0).mean()
    df.fillna('-999999', inplace=True) # N/A value treated as outlier
    #Defines forecast to predict
    forecast_column = 'Adj_Close'
    #shifts label to the past n days
    df['future_price{}d'.format(n)] = df[forecast_column].shift(-n)
    return df    
 
    
#PLOTS CORRELATION HEATMAP AND SAVES
def plot_corr(df,size=10, title = 'Correlation'):
    style.use('ggplot')
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax = sns.heatmap(corr, annot=True, cmap="Blues").set_title(title)
    fig.savefig('images/{}.png'.format(title))

    
#TRAIN LSTM 
def LSTM_train(data):
    df = data.dropna()
    X = df.drop('future_price10d', axis=1)
    y = df['future_price10d']
    X_train, X_test = X[:int(-0.1*len(X))], X[int(-0.1*(len(X))):]
    y_train, y_test = y[:int(-0.1*len(X))], y[int(-0.1*(len(X))):]
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=4, batch_size=1, verbose=2)
    score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
    print('''----- * SCORE & ACCURACY * -----\n\nScore: {}%  Accuracy: {}%
          \n\n----- * SCORE & ACCURACY * -----'''.format(round(score*100,2), round(acc*100,2)))
    return model, score, acc



#TRAIN LINEAR REGRESSION 
def LR_train(data,d):
    df = data.dropna()
    X = df.drop('future_price{}d'.format(d), axis=1)
    y = df['future_price{}d'.format(d)]
    X_train, X_test = X[:int(-0.2*len(X))], X[int(-0.2*(len(X))):]
    y_train, y_test = y[:int(-0.2*len(X))], y[int(-0.2*(len(X))):]
    reg = LinearRegression(n_jobs=-1)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print('----- * SCORE * -----\n\n{}%\n\n----- * SCORE * -----'.format(round(score*100,2)))
    return score, reg



#PREDICT FUTURE VALUES
def LR_predict(df, reg, d):
    X = df.drop('future_price{}d'.format(d), axis=1)
    y = df['future_price{}d'.format(d)]
    X_predict = X[-d:]
    reg = LinearRegression(n_jobs=-1)
    reg.fit(X[:-d], y[:-d])
    predictions = reg.predict(X_predict)
    prediction_column = [np.nan for _ in range(len(df)-d)]
    prediction_column.extend(predictions)
    df['Predictions'] = prediction_column
    return predictions, df



#PLOT STOCK PRICE VS TIME
def plot_stock(df, col, size=15, title='Stock'):
    fig = plt.figure(figsize=(size,size))
    style.use('ggplot')
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    fig.suptitle(title)
    ax1.plot(df.index, df[col])
    ax1.plot(df.index, df['20ma'])
    plt.xticks(rotation=90)
    ax2.bar(df.index, df['Adj_Volume'])
    fig.autofmt_xdate()
    plt.show()
    fig.savefig('images/{}.png'.format(title))

    
    
#PLOT PREDICTIONS
def plot_predictions(df, col, size=10, title='Stock-Predictions'):
    fig = plt.figure(figsize=(size,size))
    style.use('ggplot')
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    fig.suptitle(title)
    ax1.plot(df.index[-30:]+datetime.timedelta(days=5), df[col][-30:])
    ax1.plot(df.index[-30:], df['20ma'][-30:])
    plt.xticks(rotation=90)
    ax2.bar(df.index[-30:], df['Adj_Volume'][-30:])
    fig.autofmt_xdate()
    plt.show()
    fig.savefig('images/{}.png'.format(title))

    
    
#SENTIMENT ANALYSIS
def sentiment(text):
    tweets = TextBlob(str(text))
    return tweets.sentiment.polarity
#!!!! train lstm model !!!!


def summary(df_clean, tw, pred, days=5):
    df = df_clean.tail(len(tw)).copy()
    res = pd.concat([df, tw], axis=1, sort=True)
    part1 = res[['Adj_Close', 'Volatility', 'Change', 'Adj_Volume', '20ma']].fillna('Market Closed')
    part2 = res[['future_price{}d'.format(days), 'Predictions', 'Tweets', 'Sentiment']].fillna('Unknown')
    res = pd.concat([part1, part2], axis=1, sort=True)
    
    dates = [datetime.datetime.now().date() + datetime.timedelta(days=i) for i in range(days)]    
    predictions = pd.DataFrame(pred)
    predictions.index = dates
    predictions.columns = ['Stock-Predicted']
    return res[:-1], predictions