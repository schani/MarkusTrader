import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# from numpy.random import choice, seed
import numpy as np
# import seaborn as sns

class RandomTrader:
  def should_buy(self, history):
    return np.random.randint(0,1000) < 30
  
  # returns either False for hold, True for sell
  def should_sell(self, history, buy_date):
    return np.random.randint(0,1000) < 30

class TrendTrader:
  def should_buy(self, history):
    return history['Change'][-1]<-5
  
  def should_sell(self, history, buy_date):
    return history['Change'][-1]>5

# def trader_function(data_period, day, status):
   # if evaluation (data_period, day) == "positive": 
   #    if status== 'short' or 'hold':
           #buy(day)
# def evaluation(data, day, fundamentals)
# def sell (day):
# def buy (day):           

data = pd.read_csv('SIEmax.csv', parse_dates=['Date'], index_col='Date')
data=data.dropna()
index = pd.date_range(start= '2010-1-1', end = '2020-2-2')
print(index)

data['Change']=data.Open.pct_change().mul(100)

print(data.info())
print (data.iloc[:3])
print (data.head())
print (data.tail())
print (data.Open.iloc[0])

data.Open.plot()
plt.savefig('plot.png')
# plt.show()

def run_test(trader):
  n_days =  index.shape[0]
  start = np.random.randint(0,n_days-200) 
  # start=0
  #print (data['Change'][start])
  while not trader.should_buy(data.iloc[:start+1]):
    start=start+1
    #print (data['Change'][start])
  # time_stamp= pd.Timestamp(datetime(2019,3,3))
  # print (time_stamp.day_name())
  # data_period= pd.Period()

  buyingprice = data.Open.iloc[start]
  buyingdate = index[start]
  print('Bought at ',buyingprice, buyingdate, start)
  day=start
 

  while day<n_days-1:
    day = day+1
    history = data.iloc[:day+1]
    if trader.should_sell(history, start):
      sellingdate = index[day]
      sellingprice=data.Open.iloc[day]       
      #print ('sold at', sellingprice, sellingdate, "after %s days" %(day-start))
      pd.options.display.float_format='{:,.f2}'.format
      profit = sellingprice-buyingprice
      print (profit)
      return profit
    else:
      # hold
      pass

  #print(day)
  sellingprice=data.Open.iloc[day]
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
n=10
i=0
sells=0
profits=0
results=[]
while i<n:
  profit = run_test(TrendTrader())
  results.append(profit)
  i=i+1
  if profit !=None:
    profits= profits+profit
    sells=sells+1  
    mean_profit=profits/sells
    
print ('%s of %s times sold' % (sells, n))
print('mean profit',mean_profit) 
results.append(mean_profit)
print (results)

Results = pd.DataFrame(results, columns=['Run1'])
# print(Results)
Results.to_excel(excel_writer ='results1.xls', sheet_name = 'results1', startrow=1, startcol=2)

# results(n=100): p=0.01, 79%sold: 46.55
#p=0.001 12%sold, 78,49
#p=0.1, 100% sold, -2,82 
#(n=1000):
#p=0.1, 100% sold, -1,26
#p=0.01, 80.6%sold: 52.36
#p=0.001 15.8%sold, 107,18
#Close: 14.9%sold, 95,6 
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

