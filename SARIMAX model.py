# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:23:18 2023

@author: mrmalick
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt

import pandas as pd
from pandas_profiling import ProfileReport

#Set your working dir here
os.chdir('C:\\Sarimax')
cwd = os.getcwd()



'''
###########################################################################################################
#### ARIMA: Autoregressive Integrated Moving 
#### SARIMA: Seasonal + Autoregressive Integrated Moving Average
#### SARIMAX: SARIMA + Exogenous Variables
####Stationarity: 4 types of ways a trend can change over time.

AutoRegressive: The output is regressed on its own lagged values
Integrated: Number of times we need to do differenicng to make timeseries stationary
Moving Average: INstead of using past values, The MA model uses past forecast errors
It will determine during which period there are large changes to determine the next period value.

###########################################################################################################
'''

#get the data
data = pd.read_csv("Daily Bike Sharing.csv", 
                   index_col = "dteday", 
                   parse_dates = True)
data.head(1)

list(data.columns)


#select variables
dataset = data.loc[:, ["cnt", "holiday", "workingday", "weathersit",
                       "temp", "atemp", "hum", "windspeed"]]
dataset.head(1)   

#renaming variable
dataset = dataset.rename(columns = {'cnt' : 'y'})
dataset.head(1)

#index set freq to daily
dataset = dataset.asfreq("D")
dataset.index

#Create plot of the training data
dataset["y"].plot(figsize = (10, 7), legend = True)


"""
##STATIONARITY: DICKEY FULLER
####Stationarity: 4 types of ways a trend can change over time.
We need to do a Dickey Fuller test to see if the data is stationary. 
If it is stationary it means it has a clear well defined pattern, makes forecasting easy
We will use DIFFERENCING to make the data stationary
"""

#Stationarity
#p-value has to be less that 0.05 to consider the data stationary
from statsmodels.tsa.stattools import adfuller
stationarity = adfuller(dataset['y'])
print('Augmented Dickey Fuller p-value: %F' % stationarity[1])

"""
TRAIN AND TEST SET
"""

#Training and test set
test_days = 31
training_set = dataset.iloc[:-test_days, :]
test_set = dataset.iloc[-test_days:, :]
test_set.tail(1)

"""
EXTERNAL REGRESSORS
EXAMPLE

MOVING SEASONALITY: EVENTES LIKE BLACK FRIDAY OR HOLIDAYS THE DATE CHANGES EVERY YEAR
OUTSIDE EVENTS: 
EVENTS CAUSED BY COMPANY: INVESTMENTS THAT SHIFT THE NORMAL DEVELOPMENTS OF A KPI
"""

'''
###########################################################################################################
FORECASTING - sarimax
###########################################################################################################
'''

#exogenous variables, X-variables

#Select everything from the 2nd column until the end, the 1st column has the y variable
train_exog = training_set.iloc[:,1:]
test_exog = test_set.iloc[:,1:]
test_exog.head()


"""
3 Optimizing factors in ARIMA
p,d & q. Non negative integers, 

arima calcs these automatically

p: Order of the autoregressive, number of unknown terms that multiply signal
d: degree of first differencing, number of differes needd to make the data look stationary
q: order of the moving average part, number of unknown terms
"""

#Libraries
from pmdarima import auto_arima



"""
The m is the seasonal periods, 7 for weekly
Stepwise: 
"""
#forecasting model
model = auto_arima(y = training_set['y'],
                   X = train_exog,
                   m = 7,
                   seasonal = True,
                   stepwise = False)

model.summary()
"""
The best model parameters for p,q,d is 0,1,3
Look at the Covariance type section, this will tell you what is affecting it
hum and windspeed is negative, it decreases the sales when its more cold or windy
"""

"""
Side notes, if you run it as model.predict() it will output as an array, we want it in a series
"""

#predictions
predictions_sarimax = pd.Series(model.predict(n_periods= test_days,
                              X = test_exog)).rename("SARIMAX")
predictions_sarimax.index = test_set.index                              
predictions_sarimax

#visualization
training_set['y']['2012-07-01':].plot(figsize = (9,6), legend = True)
test_set['y'].plot(legend = True)
predictions_sarimax.plot(legend = True)


#MAE and RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(round(mean_absolute_error(test_set['y'], predictions_sarimax),0))
print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_sarimax)), 0))

#MAPE function
def MAPE(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE(test_set['y'], predictions_sarimax)


"""
If you look at the model resukts, it looks like it performed well druing the 1st half of dec but the 2nd half of dec it didnt do so well
###########################################################################################################
PROS
Easy to understand, easy implement, automated opti
CONS
Not much variables to changes, low flexibility
"""

#Export model for enseble
predictions_tbats.to_csv('predictions_sarimax.csv', index = True)









