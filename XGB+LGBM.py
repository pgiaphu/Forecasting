# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:26:30 2022

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
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import itertools

import os
username = os.getlogin()


fcperiod = 12


#sales history data
data = pd.DataFrame(pd.read_excel(r'C:\Users\{}\OneDrive - DuyTan Plastics\BI Planning\AS - FC.xlsx'.format(username),sheet_name = "B2C",index_col=0))
data.index.freq = 'MS'

#exog
#working day
wd = pd.read_excel(r'C:\Users\{}\OneDrive - DuyTan Plastics\BI Planning\WD.xlsx'.format(username),sheet_name = "Sheet1",index_col=0)
#temperature 
temperature = pd.read_excel(r'C:\Users\{}\OneDrive - DuyTan Plastics\BI Planning\Temperature.xlsx'.format(username),sheet_name = "Sheet1",index_col=0)


df_param = pd.DataFrame(data={'Model': ['XGB','LightGBM']})
df_rmse = pd.DataFrame(data={'Model': ['XGB','LightGBM']})


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
    df['tempe'] = temperature['max']
    df['tempe-lag1'] = temperature['max'].shift(1)
    df['tempe-lag2'] = temperature['max'].shift(2)
    #return df
####################################################################



#FUNCTION TO FIND OPTIMAL PARAMETER
def optimal_fc(df: pd.DataFrame()):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    y_test = y.tail(1)
    
    
    xgb_param_gridsearch = {  
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [5,10,25],
        'n_estimators': [100,1000],
        'tree_method': ['hist','exact'],
        'max_leaves': [5,10,20,40]
                        }
    
    
    xgb_all_params = [dict(zip(xgb_param_gridsearch.keys(), v)) for v in itertools.product(*xgb_param_gridsearch.values())]
    xgb_list =[]
    for params in xgb_all_params:
        xgboost = XGBRegressor(**params,objective='reg:squarederror')
        xgboost.fit(X_train, y_train)
        xgb_list.append(np.sqrt(MSE(y_train, xgboost.predict(X_train)))  )
    
    rmse_xgb = np.sqrt(MSE(y_train, xgboost.predict(X_train)))
    minxgb = xgb_all_params[np.argmin(xgb_list)]
    
    
    
    lgbm_param_gridsearch = {  
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [5,10,25],
        'n_estimators': [10,300,1000],
        #'max_leaves': [5,15,30],
        'num_leaves': [2,10,20],
        'min_gain_to_split': [3,10,20],
        'min_sum_hessian_in_leaf': [1,10]
                        }
    
    
    lgbm_all_params = [dict(zip(lgbm_param_gridsearch.keys(), v)) for v in itertools.product(*lgbm_param_gridsearch.values())]
    lgbm_list =[]
    for params in lgbm_all_params:
        lgbm = LGBMRegressor(**params)
        lgbm.fit(X_train, y_train)
        lgbm_list.append(np.sqrt(MSE(y_train, lgbm.predict(X_train)))  )
      
    rmse_lgbm = np.sqrt(MSE(y_train, lgbm.predict(X_train)))
    minlgbm = lgbm_all_params[np.argmin(lgbm_list)]
    
    return minxgb, minlgbm, rmse_xgb, rmse_lgbm





#FUNCTION TO MAKE FORECAST
####################################################################
# fit an xgboost model and make a one step prediction
def xgboost_forecast(df: pd.DataFrame(),*args):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    #y_test = y.tail(1)
    
    minxgb = {}
    for i in args:
        for key, value in i.items():
            minxgb[key] = value

    xgboost = XGBRegressor(**minxgb,objective='reg:squarederror')
    
    '''   
    param_gridsearch = {  
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [5,10,25],
        'n_estimators': [100,1000],
        'tree_method': ['hist','exact'],
        'max_leaves': [5,10,20,40]
                        }
    
    
    all_params = [dict(zip(param_gridsearch.keys(), v)) for v in itertools.product(*param_gridsearch.values())]
    xgb_list =[]
    for params in all_params:
        xgboost = XGBRegressor(**params,objective='reg:squarederror')
        xgboost.fit(X_train, y_train)
        xgb_list.append(np.sqrt(MSE(y_train, xgboost.predict(X_train)))  )
      
    minxgb = all_params[np.argmin(xgb_list)]
    '''

    
    

    # fit model
    #xgboost = XGBRegressor(n_estimators=100,max_depth=100,learning_rate=0.1,objective='reg:squarederror')
    #xgboost = XGBRegressor(n_estimators=1000)
    '''
    xgboost = BayesSearchCV(
                        XGBRegressor(solver='saga'),
                        {
                         'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                         'max_depth': Integer(1, 1000),
                         'n_estimators' : Integer(1, 1000),
                         },
                        n_iter = 20, n_jobs = -1,
                        )
    
    xgboost = RandomizedSearchCV(
                                XGBRegressor(),
                                {
                                 'learning_rate': [0.001, 0.01, 0.1, 0.5, 0.8, 1.0],
                                 'max_depth': range(1, 100),
                                 'n_estimators' : range(10,1000),
                                 },
                                n_iter=5, scoring='neg_mean_squared_error', cv=4, n_jobs = -1,verbose=1
                                    )               
    
    '''
    
    
    xgboost.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = xgboost.predict(X_test)
    #yhat = xgboost.best_estimator_.predict(X_test)
    
    return yhat[0]

def lightgbm_forecast(df: pd.DataFrame(),*args):
    X = df.iloc[:,-len(df.columns)+1:]
    y = df.iloc[:,0]
    X_train = X.head(len(X)-1)
    X_test = X.tail(1)
    y_train = y.head(len(y)-1)
    #y_test = y.tail(1)
    
    
    '''
    # fit model
    param_gridsearch = {  
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [5,10,25],
        'n_estimators': [10,300,1000],
        #'max_leaves': [5,15,30],
        'num_leaves': [2,10,20],
        #'min_gain_to_split': [0,10,20],
        'min_sum_hessian_in_leaf': [1,10]
                        }
    
    
    all_params = [dict(zip(param_gridsearch.keys(), v)) for v in itertools.product(*param_gridsearch.values())]
    lgbm_list =[]
    for params in all_params:
        lgbm = LGBMRegressor(**params)
        lgbm.fit(X_train, y_train,verbose=False)
        lgbm_list.append(np.sqrt(MSE(y_train, lgbm.predict(X_train)))  )
      
    minlgbm = all_params[np.argmin(lgbm_list)]
    '''
    
    
    minlgbm = {}
    for i in args:
        for key, value in i.items():
            minlgbm[key] = value
    
    lgbm = LGBMRegressor(**minlgbm)
    
    
    
    
    #lgbm = LGBMRegressor(n_estimators=100,max_depth=20,learning_rate=0.01,num_leaves=100,min_gain_to_split=50)

   
    '''
    lgbm =  RandomizedSearchCV(
                        LGBMRegressor(),
                        {
                         'learning_rate': Real(10e-6, 1.0,prior='log-uniform'),
                         'max_depth': Integer(0, 100),
                         #'num_leaves': Integer(1,10,prior='log-uniform'),
                         #'min_gain_to_split': Real(0.0,0.9,prior='log-uniform'),
                         'n_estimators' : Integer(1, 100),
                         #'min_sum_hessian_in_leaf' :Integer(1,10,prior='log-uniform')
                         },
                        n_jobs=-1
                        #n_iter = 25, scoring = 'roc_auc', error_score = 0, verbose = 3, n_jobs = -1,
                        )
    
    lgbm =  BayesSearchCV(
                        LGBMRegressor(),
                        {
                         'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
                         'max_depth': Integer(0, 50,prior='log-uniform'),
                         'num_leaves': Integer(1,50,prior='log-uniform'),
                         'n_estimators' : Integer(1, 100),
                         },
                        n_jobs=-1,n_iter = 10, scoring = 'roc_auc'
                        )
    

    '''
   

   
    
    lgbm.fit(X_train, y_train,verbose=False)
    # make a one-step prediction
    yhat = lgbm.predict(X_test)
    #yhat = lgbm.best_estimator_.predict(X_test)
    return yhat[0]

####################################################################


df_XGB = pd.DataFrame()
df_LGBM = pd.DataFrame()
df_fc = pd.DataFrame()
df_best = pd.DataFrame()

for sku in list(data):
    df = pd.DataFrame(data[sku].copy(deep=True))
    minxgb, minlgbm, rmse_xgb, rmse_lgbm = optimal_fc(df)
    df_param[sku] = np.nan
    df_param[sku] = df_param[sku].astype('object')
    df_param.at[df_param[df_param['Model'] == 'XGB'].index[0],sku] = ( minxgb  )
    df_param.at[df_param[df_param['Model'] == 'LightGBM'].index[0],sku] = ( minlgbm  )
    
    df_rmse[sku] = np.nan
    df_rmse[sku] = df_rmse[sku].astype('object')
    df_rmse.at[df_param[df_rmse['Model'] == 'XGB'].index[0],sku] = rmse_xgb
    df_rmse.at[df_param[df_rmse['Model'] == 'LightGBM'].index[0],sku] = rmse_lgbm
    

    for i in range(1,fcperiod+1):
        if i == 1:
            df_fc = pd.concat([df,make_future_dataframe(df,1)])
        else:
            df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
        time_features(df_fc)
        df_fc.iloc[-1:,0] = lightgbm_forecast(df_fc,minlgbm)
    df_LGBM[sku] = df_fc[sku].tail(fcperiod)
        
     
    for i in range(1,fcperiod+1):
        if i == 1:
            df_fc = pd.concat([df,make_future_dataframe(df,1)])
        else:
            df_fc = pd.concat([df_fc,make_future_dataframe(df_fc,1)])
        time_features(df_fc)
        df_fc.iloc[-1:,0] = xgboost_forecast(df_fc,minxgb)
    df_XGB[sku] = df_fc[sku].tail(fcperiod)
    
    if rmse_lgbm < rmse_xgb:
        df_best[sku] = df_LGBM[sku]
    else:
        df_best[sku] = df_XGB[sku]
        
    
df_XGB['Model'] = 'XGB'
df_LGBM['Model'] = 'LightGBM'



pd.concat([df_XGB,df_LGBM],axis=0).to_excel(r'C:\Users\{}\Downloads\MLFC_Full.xlsx'.format(username))
df_param.to_excel(r'C:\Users\{}\Downloads\MLFC_Params.xlsx'.format(username))
df_rmse.to_excel(r'C:\Users\{}\Downloads\MLFC_RMSE.xlsx'.format(username))
df_best.to_excel(r'C:\Users\{}\Downloads\MLFC.xlsx'.format(username))


'''
fig, ax = plt.subplots()
ax.plot(df[sku], color='k', label='Actual')
ax.plot(df_LGBM[sku], label='LGBM')
ax.plot(df_XGB[sku], label='XGB')
plt.legend()
plt.show()
'''








