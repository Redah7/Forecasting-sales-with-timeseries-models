# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:45:18 2023

@author: mrmalick
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\mrmalick\\OneDrive - Shoprite Checkers (Pty) Limited\\Desktop\\Learning\\Forecasting Models and Time Series for Business in Python\\4. TBATS')
cwd = os.getcwd()

#get the data
data = pd.read_csv("Daily Bike Sharing.csv", 
                   index_col = "dteday", 
                   parse_dates = True)
data.head(1)

'''
###########################################################################################################
#select variables
###########################################################################################################
'''

dataset = data.loc[:, ["cnt", "holiday", "workingday", "weathersit",
                       "temp", "atemp", "hum", "windspeed"]]
dataset.head(1)    

'''
###########################################################################################################
#renaming variable, set index
###########################################################################################################
'''

dataset = dataset.rename(columns = {'cnt' : 'y'})
dataset.head(1)

#index
dataset = dataset.asfreq("D")
dataset.index

'''
###########################################################################################################
DATA VIZ
###########################################################################################################
'''

dataset["y"].plot(figsize = (10, 7), legend = True)

'''
###########################################################################################################
#Training and test set
###########################################################################################################
'''
#we will use the last 31 days as the testing set

test_days = 31
#[rows, columns] we wants all rows from start until the last 31 days (-31)
training_set = dataset.iloc[:-test_days, :]
#check that the last rows of data should not be in Dec
training_set.tail(2)
#We test on from the last 31 days to the end of the rows
test_set = dataset.iloc[-test_days:, :]
test_set.tail(1)


'''
###########################################################################################################
Model - 

ARMA errors: 
1. AutoRegressive component
Uses previous numbers to predict future. The latest values holds more value and power.
It uses LAGS.

2. Moving average: It creates a visual for the errors, corrects the predictions as it deviates 
from the error. 

3. Trig seasonality: dont need to know how it applied

4. box cox transformations: transforms the dependant variables into a normal distribution.


###########################################################################################################
'''
#TBATS

from tbats import TBATS

###########################################################################################################
#predictions

#use_trend if you know there's a seasonal trend, set it to true.
#seasonal_periods [weekly pattern, yearly pattern], we have a leap year so about 365.5 days

model = TBATS(use_trend=True , seasonal_periods=[7,365.5])
model = model.fit(training_set['y'])     



#Set preds, note: output is an array, turn into series with an outer wrapper
predictions_tbats = pd.Series(model.forecast(steps = len(test_set))).rename("TBATS")
#apply the index to the series, the prediction set will have the same index as the test set
predictions_tbats.index = test_set.index
#check that you have a series with index and value
predictions_tbats.head()


###########################################################################################################
#visualization
training_set['y']['2012-07-01':].plot(figsize = (9,6), legend = True)
test_set['y'].plot(legend = True)
predictions_tbats.plot(legend = True)
#The results show now seasonality, just a straight line

'''
###########################################################################################################
Assessment
###########################################################################################################
'''
#MAE and RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(round(mean_absolute_error(test_set['y'], predictions_tbats),0))
print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_tbats)), 0))


#MAPE function
def MAPE(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE(test_set['y'], predictions_tbats)

#Export
predictions_tbats.to_csv('predictions_tbats.csv', index = True)










