# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:43:54 2023

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


os.chdir('C:\\Users\\mrmalick\\OneDrive - Shoprite Checkers (Pty) Limited\\Desktop\\Learning\\Forecasting Models and Time Series for Business in Python\\6. Tensorflow Probabilities Structural Time Series')
cwd = os.getcwd()


'''
###########################################################################################################
THEORY
###########################################################################################################
For every timeseries data you decompose: there's trend, seasonality & exogenous factors/impacts and then noise/error
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

#index
dataset = dataset.asfreq("D")
dataset.index


#viz
dataset["y"].plot(figsize = (10, 7), legend = True)



"""
TRAIN AND TEST SET
"""

#Training and test set
#testing on the last 31 days
test_days = 31
training_set = dataset.iloc[:-test_days, :]
test_set = dataset.iloc[-test_days:, :]
test_set.tail(1)


'''
###########################################################################################################
THEORY - DECOMPOSE
###########################################################################################################
TREND, SEASONALITY, EXOG IMPACTS, AUTOREGRESSIVE, NOISE

SEASONALITY: WEEKLY, MONTHLY, YEARLY, THIS IS WHY WE DECLARE IT
AUTOREGRESSIVE: WE GIVE A LARGER WEIGHT TO THE LATEST/MOST RECENT DATA
EXOG IMPACTS: LINEAR REGRESSION IS CREATED TO ASSESS IMPACT OF EACH INDIVDUALLY

'''

'''
###########################################################################################################
THEORY - REGRESSORS
###########################################################################################################
'''

#get a library
import tensorflow_probability as tfp

'''
###########################################################################################################
#Isolate the regressors
###########################################################################################################
'''

#Give all rows for columns from the 2nd to the end
#exog = dataset.iloc[:, 1:]

#WE NEED to transfer the data into a matrix
#the some variables are integers (0,1,2). Wee need it into floats ie decimals, we add the astype to convert
#this is the design matrix
exog = np.asmatrix(dataset.iloc[:,1:].astype(np.float64))
#the .float changed the integers from 2 to 2.0
exog[:1]

'''
###########################################################################################################
#linear regression
###########################################################################################################
'''

#sts is Structural TimeSeries
regressors = tfp.sts.LinearRegression(design_matrix = exog,
                                      name = "regressors")



'''
###########################################################################################################
#SEASONALITY - #isolating dependent variable
 ###########################################################################################################
'''
#isolating dependent variable
#
y = training_set['y'].astype(np.float64)
y[:5]


"""
#Weeklday SEASONALITY - #isolating dependent variable
"""

#Our data is daily so we need to undersatnd the weekday effect, & days a week in steps of `1`
#Read up on the docuemntation for this model
#tfp = Tensor Flow Probabilities
weekday_effect = tfp.sts.Seasonal(num_seasons = 7,
                                  num_steps_per_season = 1,
                                  observed_time_series = y,
                                  name = "weekday_effect")


"""
#MONTHLY SEASONALITY - #isolating dependent variable
#Need to take car of leap year, 2012 had 29 days in Feb
#number of days per month
"""

num_days_per_month = np.array(
    [[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], #2011
     [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]) # 2012

#Monthly seasonality, the 
monthly_effect = tfp.sts.Seasonal(num_seasons = 12,
                                  num_steps_per_season = num_days_per_month,
                                  observed_time_series = y,
                                  name = "monthly_effect")


"""
AUTOREGRESSIVE AND TREND COMPONENTS
"""
#trend 
trend = tfp.sts.LocalLinearTrend(observed_time_series=y,
                                 name = "trend")


#Autoregressive
autoregressive = tfp.sts.Autoregressive(order = 1,
                                        observed_time_series = y,
                                        name = "autoregressive")


"""
#Tensorflow Structural Time series
"""

#forecasting model, the components added as x variables
model = tfp.sts.Sum([regressors,
                     weekday_effect,
                     monthly_effect,
                     autoregressive,
                     trend],
                    observed_time_series = y)


#fit with HMC(Hamiltonian Monte Carlo)
#The casual inference: We know what happenend but we dont know what led to it

"""
THIS MODEL FIT WILL TAKE VERY LONG
"""


"""
num_results, FOR EVERY PREDICTION PROVIDE 100 RESULTS
num_warmup_steps: 

"""

samples, kernel_results = tfp.sts.fit_with_hmc(model = model,
                                               observed_time_series = y,
                                               num_results = 100,
                                               num_warmup_steps = 50,
                                               num_leapfrog_steps = 15,
                                               num_variational_steps = 150,
                                               seed = 1501)

#forecast
forecast = tfp.sts.forecast(model = model,
                            observed_time_series = y,
                            parameter_samples = samples,
                            num_steps_forecast = len(test_set))


#format the predictions
predictions_tfp = pd.Series(forecast.mean()[:,0], name = "TFP")
predictions_tfp.index = test_set.index
predictions_tfp[:2]

"""
Could add a dummy variable to highlight christmas season to improve accuracy
"""

#visualization
training_set['y']['2012-07-01':].plot(figsize = (9,6), legend = True)
test_set['y'].plot(legend = True)
predictions_tfp.plot(legend = True)


#MAE and RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(round(mean_absolute_error(test_set['y'], predictions_tfp),0))
print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_tfp)), 0))


#MAPE function
def MAPE(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE(test_set['y'], predictions_tfp)

"""
Pro's and cons'

Pro: flexible, egressors, good with short term dynamics
Cons: complex programming, slow model, 
"""

predictions_tfp.to_csv('predictions_tfp.csv', index = True)











