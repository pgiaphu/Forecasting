# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:19:23 2022

@author: PhamGiaPhu
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

fcperiod = 12

#sales history data
data = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\AS - Prophet.xlsx',index_col=0))
data.index.freq = 'MS'

#exog
wd = pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Master Data\WD.xlsx',sheet_name = "Sheet1",index_col=0)
#exog for fitting
exog_fit = df.merge(wd[['WD']],left_index=True,right_index=True,how='inner')
exog_fit = exog_fit.drop(exog_fit.columns.difference(['WD']),axis=1) #drop other column
#exog for forecast
exog_fc = wd.merge(df,left_index=True,right_index=True,how='outer',indicator=True).query('_merge == "left_only"') #anti left join
exog_fc = exog_fc.drop(exog_fc.columns.difference(['WD']),axis=1).head(fcperiod)



#PRE-FORECAST FUNCTION
####################################################################
#make dataframe for walk-forward forecast
def make_future_dataframe(self, periods, freq='MS'):
    #make dataframe for forecast
    last_date = self.index.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq)
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods
    return pd.DataFrame(index= dates)

#feature enginering    
def time_features(df: pd.DataFrame()):
    #Creates time series features from datetime index
    df['date'] = pd.to_numeric(df.index, downcast='integer')
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag1'] = df.iloc[:,0].shift(1)
    df['lag2'] = df.iloc[:,0].shift(2)
    df['lag3'] = df.iloc[:,0].shift(3)
    df['lag4'] = df.iloc[:,0].shift(4)
    df['wd'] = wd.WD
    df['wd-lag1'] = wd.WD.shift(1)
    df['wd-lag2'] = wd.WD.shift(2)
    #return df
####################################################################




#FUNCTION TO MAKE FORECAST
####################################################################
# fit an xgboost model and make a one step prediction
def xgboost_forecast(df: pd.DataFrame()):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    y_test = y.tail(1)

    # fit model
    xgboost = XGBRegressor(n_estimators=1000,max_depth=20,learning_rate=0.1,objective='reg:squarederror')
    xgboost.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = xgboost.predict(X_test)
    return yhat[0]

def lightgbm_forecast(df: pd.DataFrame()):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    y_test = y.tail(1)
    
    # fit model
    lgbm = LGBMRegressor(n_estimators=100,max_depth=20,learning_rate=0.1,num_leaves=10,min_gain_to_split=20)
    """
    lgbm =  RandomizedSearchCV(
                        LGBMRegressor(),
                        {
                         'learning_rate': Real(0.001, 1.0,prior='log-uniform'),
                         'max_depth': Integer(1, 100,prior='log-uniform'),
                         #'num_leaves': Integer(1,10,prior='log-uniform'),
                         #'min_gain_to_split': Real(0.0,0.9,prior='log-uniform'),
                         'n_estimators' : Integer(1, 100,prior='log-uniform'),
                         #'min_sum_hessian_in_leaf' :Integer(1,10,prior='log-uniform')
                         }
                        )
   
    """
    lgbm.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = lgbm.predict(X_test)
    #yhat = lgbm.best_estimator_.predict(X_test)
    return yhat[0]

####################################################################
df_XGB = pd.DataFrame()
df_LGBM = pd.DataFrame()
df_fc = pd.DataFrame()

for sku in list(data):
    df = pd.DataFrame(data[sku].copy(deep=True))

    for i in range(1,fcperiod+1):
        if i == 1:
            df_fc = pd.concat([df,make_future_dataframe(df,1)])
        else:
            df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
        time_features(df_fc)
        df_fc.iloc[-1:,0] = lightgbm_forecast(df_fc)
    df_LGBM[sku] = df_fc[sku].tail(fcperiod)
        
     
    for i in range(1,fcperiod+1):
        if i == 1:
            df_fc = pd.concat([df,make_future_dataframe(df,1)])
        else:
            df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
        time_features(df_fc)
        df_fc.iloc[-1:,0] = xgboost_forecast(df_fc)
    df_XGB[sku] = df_fc[sku].tail(fcperiod)


'''
fig, ax = plt.subplots()
ax.plot(df[sku], color='k', label='Actual')
ax.plot(df_LGBM[sku], label='LGBM')
ax.plot(df_XGB[sku], label='XGB')
plt.legend()
plt.show()
'''
