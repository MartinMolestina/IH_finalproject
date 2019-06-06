import statistics
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np

# MARKOV CHAIN MONTE CARLO SIMULATION

def mcmc(df, days=1000, ticker='COMPANY'):
    df['change_ratio'] = (df['close']-df['open'])/df['open']
    stats = df['change_ratio'].describe()
    mu, sigma = stats['mean'], stats['std']

    # plots stock's change histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    style.use('ggplot')
    sns.distplot(df['change_ratio'])
    ax.set_xlabel('Change / OpenPrice')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of {} stock change ratio:\n $\mu={}$, $\sigma={}$'.format(ticker,round(mu,7),round(sigma,3)))
    fig.savefig('images/Histogram of {} stock change ratio: mu={}, sigma={}.png'.format(ticker,round(mu,7),round(sigma,3)))
    
    # Montecarlo simulation
    sample = mu + sigma * np.random.randn()
    tomorrow = df['close'][-1] + df['close'][-1]*sample
    res = []
    end = []
    for scene in range(500):
        evo = []
        future = tomorrow
        for day in range(days):
            sample = mu + sigma * np.random.randn()
            future += future*sample
            evo.append(future)
        res.append(evo)
        end.append(evo[-1])
    results = pd.DataFrame(res).T
    
    # Plot MCMC simulations
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(results, alpha=0.4)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Stock Price ')
    plt.title('MCMC of {} stock price'.format(ticker))
    plt.show()
    fig.savefig('images/MCMC of {} stock price.png'.format(ticker))
    
    #Plot results histogram
    mean = statistics.mean(end)
    std = statistics.stdev(end)
    fig, ax = plt.subplots(figsize=(10, 5))
    style.use('ggplot')
    sns.distplot(end)
    ax.set_xlabel('Price {} days into the future'.format(days))
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of {} future price :\n $\mu={}$, $\sigma={}$'.format(ticker,round(mean,2),round(std,2)))
    plt.show()
    fig.savefig('images/Histogram of {} future price :mu={},sigma={}.png'.format(ticker,round(mean,2),round(std,2)))
    
    # Probability table
    table = []
    for i in range(10):
        table.append((round(1-i/10,2), scipy.stats.norm.ppf(i/10,mean,std)))

    prob_table = pd.DataFrame(table)
    prob_table.columns = ['Probability', 'Min. Price']
    prob_table.set_index('Probability', inplace=True)
    print(prob_table)

    return prob_table