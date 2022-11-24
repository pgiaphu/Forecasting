# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:25:06 2022
# -*- coding: utf-8 -*-
@author: PhamGiaPhu
"""

import itertools
from prophet import Prophet
import pmdarima as pmd
import pandas as pd 
import numpy as np
from scipy.stats import zscore
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import datetime as dt

df = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\AS - Prophet.xlsx',index_col=0))
#df = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\ASB2Bbtm.xlsx',index_col=0))
#df = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\AS - B2C Month aT.xlsx',index_col=0))

#fill nan with median of to prepare data for outlier detection
df = df.fillna(df.median())
df = df[(np.abs(df.apply(zscore))<2.3)]
df = df.fillna(df.median())
wd = pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Master Data\WD.xlsx',sheet_name = "Sheet1",index_col=0)
#forecast period
fcperiod = 12


#create list of forecast date
future_index = []
future_index.append(df.tail(fcperiod).index.shift(fcperiod,freq="MS"))

#exog for fitting
exog_fit = df.merge(wd[['WD']],left_index=True,right_index=True,how='inner')
exog_fit = exog_fit.drop(exog_fit.columns.difference(['WD']),axis=1) #drop other column

#exog for forecast
exog_fc = wd.merge(df,left_index=True,right_index=True,how='outer',indicator=True).query('_merge == "left_only"') #anti left join
exog_fc = exog_fc.drop(exog_fc.columns.difference(['WD']),axis=1).head(fcperiod)


firstsku = list(df)[0]
df_SES = pd.DataFrame()
df_P = pd.DataFrame()
df_HW = pd.DataFrame()
df_UCM = pd.DataFrame()
df_SARIMA = pd.DataFrame()
df_param = pd.DataFrame(data={'Model': ['SES','Holt-Winter','SARIMAX','UCM']})


for sku in list(df):
    try:
        #Prophet
        param_gridsearch = {  
            'changepoint_prior_scale': [0.1, 0.5],
            'growth': ['logistic','linear','flat'],
            'seasonality_prior_scale': [1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [3,6],
            'changepoint_range': [0.95],
                            }
        all_params = [dict(zip(param_gridsearch.keys(), v)) for v in itertools.product(*param_gridsearch.values())]
        rmses = []
        cutoffs = pd.date_range(start='2022-05-01', end='2022-07-01', freq='MS') 
        
        
        df_model = pd.DataFrame(df[sku].copy()).reset_index()
        df_model.rename(columns={'Date': 'ds', sku: 'y'},inplace=True)
        df_model['cap'] = df_model.y.quantile(0.95)+1
        df_model['floor'] = df_model.y.quantile(0.1)
        df_model['wd'] = np.asarray(exog_fit)
         
        m = (
            Prophet(growth='logistic',
                    changepoint_prior_scale=0.2,uncertainty_samples=0,
                    weekly_seasonality=False,daily_seasonality=False,
                    seasonality_mode='additive')
                    .add_seasonality(name='monthly', period=12, 
                                     fourier_order=12,prior_scale=0.4).
                    add_regressor('wd')
                    .fit(df_model))
        if sku == firstsku:
            df_f = m.make_future_dataframe(periods=fcperiod,freq='MS')
        else:
            pass
        df_f['cap'] = df_model.y.quantile(0.95)+1
        df_f['floor'] = df_model.y.quantile(0.1)
        df_f['wd'] = np.asarray(np.concatenate((exog_fit.to_numpy(),exog_fc.to_numpy()),axis=0))
        forecast = m.predict(df_f)
        df_P[sku] = forecast.yhat.tail(fcperiod)
        #fig = m.plot(forecast)
    
        #SES model    
        fitSES = sm.tsa.SimpleExpSmoothing(np.asarray(df[sku])).fit(optimized=True)
        arr_forecast = fitSES.forecast(fcperiod)
        df_SES[sku] = arr_forecast
        df_param[sku] = np.nan
        df_param[sku] = df_param[sku].astype('object')
        df_param.at[df_param[df_param['Model'] == 'SES'].index[0],sku] = ( fitSES.params.get('smoothing_level')  )
    
        #Holt-Winter model
        HW_param_gridsearch = {  
            'initialization_method': ['heuristic','estimated','legacy-heuristic'],
            'seasonal': ['add','mul'],
            'trend': ['add','mul'],
            'damped_trend': [True,False],
            'use_boxcox': [True,False],
                            }
        HW_all_params = [dict(zip(HW_param_gridsearch.keys(), v)) for v in itertools.product(*HW_param_gridsearch.values())]
        hw =[]
        for params in HW_all_params:
            hw.append(sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), seasonal_periods=12,**params).fit(optimized=True).aicc)
        minhw = HW_all_params[np.argmin(hw)]
        fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), seasonal_periods=12,**minhw).fit(optimized=True)
       #fitHW = sm.tsa.ExponentialSmoothing(np.asarray(df[sku]), initialization_method='heuristic',seasonal_periods=12,trend='add', seasonal='add',damped_trend=True).fit(optimized=True)
 
        arr_forecast = fitHW.forecast(fcperiod)
        df_HW[sku] = arr_forecast 
        df_param.at[df_param[df_param['Model'] == 'Holt-Winter'].index[0],sku] = ( minhw.update(fitHW.params)  )
        
        
        
        #df_HW.set_index(future_index,inplace=True)
        #df.append(df_HW).plot()
        
    
        #SARIMA  
        #check for how many diffs we need
        d = pmd.arima.ndiffs(np.asarray(df[sku])) #first diff
        D = pmd.arima.nsdiffs(np.asarray(df[sku]), 12) #seasonal diff
        ap_autoarimamodel = pmd.arima.auto_arima(np.asarray(df[sku]),
                                                 information_criterion = 'aicc',
                                                 exogenous=exog_fit,
                                                 start_p=0, max_p=12,
                                                 d=d, max_d=d,
                                                 start_q=0, max_q=12,
                                                 start_P=0, max_P=3,
                                                 start_Q=0, max_Q=3,
                                                 D=D,max_D=D,
                                                 m=12,seasonal=True,
                                                 trace=False,supress_warnings=True,stepwise=True,error_action='ignore',random_state=20,n_fits=50)

        df_param.at[df_param[df_param['Model'] == 'SARIMAX'].index[0],sku] = ap_autoarimamodel.get_params().get("order") + ap_autoarimamodel.get_params().get("seasonal_order")
        
        try:
           # arr_forecast = ap_autoarimamodel.predict(n_periods=fcperiod,
           #                                          exogenous=exog_fc,
           #                                          return_conf_int = False)
            df_SARIMA[sku] = ap_autoarimamodel.predict(n_periods=fcperiod,
                                                       exogenous=exog_fc,
                                                       return_conf_int = False)
        except:
            pass
    
        
        #UCM model
        #grid search
        UCM_param_gridsearch = {  
            'level': ['ntrend','dconstant','llevel','rwalk','dtrend','lldtrend','rwdrift','lltrend','strend','rtrend'],
            'cycle': [True,False],
            'irregular': [True,False],
            'damped_cycle': [True,False],
            'use_exact_diffuse': [True,False],
            'autoregressive': [1,2]
                            }
        UCM_all_params = [dict(zip(UCM_param_gridsearch.keys(), v)) for v in itertools.product(*UCM_param_gridsearch.values())]
        aicc =[]
        for params in UCM_all_params: 
               aicc.append(
                   sm.tsa.UnobservedComponents(
                   np.asarray(df[sku]),
                   exog = exog_fit,
                   **params,
                   freq_seasonal=[{'period':12,'harmonics':12}]).fit().aicc)
        minaicc = UCM_all_params[np.argmin(aicc)]

        fitUCM = sm.tsa.UnobservedComponents(
            np.asarray(df[sku]),
            exog = exog_fit,
            level= minaicc,
            cycle=True,irregular=True,damped_cycle=True,
            freq_seasonal=[{'period':12,'harmonics':12}]).fit()
        arr_forecast = fitUCM.forecast(fcperiod,exog = exog_fc)
        df_UCM[sku] = arr_forecast
        df_param.at[df_param[df_param['Model'] == 'UCM'].index[0],sku] = ( minaicc  )
        #df_UCM.set_index(future_index,inplace=True)
    except:
        pass

df_P.set_index(future_index,inplace=True)
df_P['Model'] = 'Prophet'
df_SES.set_index(future_index,inplace=True)
df_SES['Model'] = 'SES'
df_HW.set_index(future_index,inplace=True)
df_HW['Model'] = 'Holt-Winter'
df_UCM.set_index(future_index,inplace=True)
df_UCM['Model'] = 'UCM'
df_SARIMA.set_index(future_index,inplace=True)
df_SARIMA['Model'] = 'SARIMA'

#comparing model and select best fit
aiccmodel = {'SES': fitSES.aicc,
             'Holt-Winter': fitHW.aicc,
             'SARIMAX': ap_autoarimamodel.aicc(),
             'UCM': fitUCM.aicc
             }


pd.concat([df_SES,df_P,df_HW,df_UCM,df_SARIMA],axis=0).to_excel('FC B2C Weekly Adjust.xlsx')
df_param.to_excel('Param.xlsx')
