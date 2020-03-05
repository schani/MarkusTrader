import pandas as pd
# from datetime import datetime
# import matplotlib.pyplot as plt
# from numpy.random import choice, seed
import numpy as np
# import seaborn as sns

# returns either False for hold, True for sell
def random_eval(history, buy_date):
  # print(history)
  return np.random.randint(0,100) < 10

# def trader_function(data_period, day, status):
   # if evaluation (data_period, day) == "positive": 
   #    if status== 'short' or 'hold':
           #buy(day)
# def evaluation(data, day, fundamentals)
# def sell (day):
# def buy (day):           
data = pd.read_csv('TSLA.csv', parse_dates=['Date']) #index_col='Date')

#print(data.info())

def run_test(evaluator):
  n_days =  data.shape[0]

  start = np.random.randint(0,n_days-100) 
  # time_stamp= pd.Timestamp(datetime(2019,3,3))
  # print (time_stamp.day_name())
  # data_period= pd.Period()

  buyingprice = data['Open'][start]
  buyingdate = data['Date'][start]
  print('Bought at ',buyingprice, buyingdate, start)
  day=start
  partial_data= pd.date_range(start=data['Date'][0], end= data['Date'][day])
  #print (partial_data)

  while day<n_days-1:
    day = day+1
    history = data[:day+1]
    if evaluator(history, start):
      sellingdate = data['Date'][day]
      sellingprice=data['Open'][day]       
      print ('sold at', sellingprice, sellingdate, "after %s days" %(day-start))
      pd.options.display.float_format='{:,.f2}'.format
      profit = sellingprice-buyingprice
      print (profit)
      return profit
    else:
      # hold
      pass

  #print(day)
  sellingprice=data['Open'][day]
  print('hold at ',sellingprice, day)
  pd.options.display.float_format='{:,.f2}'.format
  return None

# TODO fuer Markus:
# 1. Eine Schleife machen, die `run_test()` 1000 mal oder so
#    laufen laesst, und die Profite (wenn != None) mittelt und
#    den erwarteten Profit ausgibt
# 2. Stop-loss evaluator schreiben.
# 3. Random vs Stop-loss vergleichen.
# 4. Andere Daten ausprobieren.

run_test(random_eval)

#data.plot(subplots=True)
#plt.show()

# Set seed here
#seed(42)

# Calculate daily_returns here
#daily_returns = data['Open'].pct_change().dropna()

# Get n_obs
#n_obs = daily_returns.count()

# Create random_walk
#random_walk = choice(daily_returns, size=n_obs)

# Convert random_walk to pd.series
#random_walk = pd.Series(random_walk)

# Plot random_walk distribution
#sns.distplot(random_walk)
#plt.show()

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

