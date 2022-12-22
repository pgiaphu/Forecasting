# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:25:06 2022
# -*- coding: utf-8 -*-
@author: PhamGiaPhu
"""

import itertools
from prophet import Prophet
from sklearn.metrics import mean_squared_error as MSE
import pmdarima as pmd
import pandas as pd 
import numpy as np
from scipy.stats import zscore
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import datetime as dt
from sklearn.metrics import mean_squared_error
import os
username = os.getlogin()



df = pd.DataFrame(pd.read_excel(r'C:\Users\{}\OneDrive - DuyTan Plastics\BI Planning\AS - FC.xlsx'.format(username),sheet_name = "B2C",index_col=0))
#df = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\ASB2Bbtm.xlsx',index_col=0))
#df = pd.DataFrame(pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Script\AS - B2C Month aT.xlsx',index_col=0))

#fill nan with median of to prepare data for outlier detection
df = df.fillna(df.median())
df = df[(np.abs(df.apply(zscore))<2.5)]
df = df.fillna(df.median())
wd = pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Master Data\WD.xlsx',sheet_name = "Sheet1",index_col=0)

tempe = pd.read_excel(r'C:\Users\PhamGiaPhu\OneDrive - DuyTan Plastics\Master Data\Temperature.xlsx',sheet_name = "Sheet1",index_col=0)



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
df_best = pd.DataFrame()
df_param = pd.DataFrame(data={'Model': ['Prophet','SES','Holt-Winter','SARIMAX','UCM']})
df_rmse = pd.DataFrame(data={'Model': ['Prophet','SES','Holt-Winter','SARIMAX','UCM']})

for sku in list(df):
    try:
        #Prophet

        param_gridsearch = {  
            'changepoint_prior_scale': [0.1, 0.5],
            'growth': ['logistic','linear'],
            #'seasonality_prior_scale': [0.1, 4],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [3],
                            }
        
        
        all_params = [dict(zip(param_gridsearch.keys(), v)) for v in itertools.product(*param_gridsearch.values())]
        rmses = []

        
        
        df_model = pd.DataFrame(df[sku].copy()).reset_index()
        df_model.rename(columns={'Date': 'ds', sku: 'y'},inplace=True)
        df_model['cap'] = df_model.y.quantile(0.95)+1
        df_model['floor'] = df_model.y.quantile(0.1)
        df_model['wd'] = np.asarray(exog_fit)
         
        for params in all_params:
            #cross validation search for best fit
            m = (
                Prophet(**params,weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
                        .add_seasonality(name='monthly', period=12, fourier_order=12,prior_scale=0.1)
                        .add_regressor('wd')
                        .add_country_holidays(country_name='VN')
                        .fit(df_model))

            rmses.append(np.sqrt(MSE(df_model['y'], m.predict(df_model)['yhat'] )))
            
        best_params = all_params[np.argmin(rmses)]   
        m = (
            Prophet(**best_params,weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
                    .add_seasonality(name='monthly', period=12, fourier_order=12,prior_scale=0.1)
                    .add_regressor('wd')
                    .add_country_holidays(country_name='VN')
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
        df_param[sku] = np.nan
        df_rmse[sku] = np.nan
        df_param[sku] = df_param[sku].astype('object')
        df_rmse[sku] = df_param[sku].astype('object')
        df_param.at[df_param[df_param['Model'] == 'Prophet'].index[0],sku] = ( best_params  )
        df_rmse.at[df_rmse[df_rmse['Model'] == 'Prophet'].index[0],sku]  = np.sqrt(MSE(df_model['y'], m.predict(df_model)['yhat'] ))
    
        #SES model    
        fitSES = sm.tsa.SimpleExpSmoothing(np.asarray(df[sku])).fit(optimized=True)
        df_SES[sku] = fitSES.forecast(fcperiod)
        df_param.at[df_param[df_param['Model'] == 'SES'].index[0],sku] = ( fitSES.params.get('smoothing_level')  )
        df_rmse.at[df_rmse[df_rmse['Model'] == 'SES'].index[0],sku] = mean_squared_error(np.asarray(df[sku]), fitSES.fittedvalues, squared=False)
    
        #Holt-Winter model
        HW_param_gridsearch = {  
            'initialization_method': ['heuristic','estimated'],
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
 
        df_HW[sku] = fitHW.forecast(fcperiod)
        df_param.at[df_param[df_param['Model'] == 'Holt-Winter'].index[0],sku] = ( minhw )
        df_rmse.at[df_rmse[df_rmse['Model'] == 'Holt-Winter'].index[0],sku] = mean_squared_error(np.asarray(df[sku]), fitHW.fittedvalues, squared=False)
        
        
        
        #df_HW.set_index(future_index,inplace=True)
        #df.append(df_HW).plot()
        
    
        #SARIMA  
        #check for how many diffs we need
        ap_autoarimamodel = pmd.arima.auto_arima(np.asarray(df[sku]),
                                                 information_criterion = 'aicc',
                                                 exogenous=exog_fit,
                                                 start_p=0, max_p=5,
                                                 d=1, max_d=1,
                                                 start_q=0, max_q=5,
                                                 start_P=0, max_P=3,
                                                 start_Q=0, max_Q=3,
                                                 D=1,max_D=1,
                                                 m=12,seasonal=True,
                                                 trace=False,supress_warnings=True,stepwise=True,with_intercept=True,error_action='ignore',random_state=25,n_fits=50)

        df_param.at[df_param[df_param['Model'] == 'SARIMAX'].index[0],sku] = ap_autoarimamodel.get_params().get("order") + ap_autoarimamodel.get_params().get("seasonal_order")
        df_rmse.at[df_rmse[df_rmse['Model'] == 'SARIMAX'].index[0],sku] = mean_squared_error(np.asarray(df[sku]), ap_autoarimamodel.arima_res_.fittedvalues, squared=False)
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
            #'level': ['ntrend','dconstant','llevel','rwalk','dtrend','lldtrend','rwdrift','lltrend','strend','rtrend'],
            'level': ['ntrend','lldtrend','lltrend','strend','rtrend'],
            'cycle': [True,False],
            'irregular': [True,False],
            'damped_cycle': [True,False],
            'use_exact_diffuse': [True,False],
            'autoregressive': [0,1]
                            }
        UCM_all_params = [dict(zip(UCM_param_gridsearch.keys(), v)) for v in itertools.product(*UCM_param_gridsearch.values())]
        aicc =[]
        for params in UCM_all_params: 
               try:
                   aicc.append(
                       sm.tsa.UnobservedComponents(
                       np.asarray(df[sku]),
                       exog = exog_fit,
                       **params,
                       freq_seasonal=[{'period':12,'harmonics':12}]).fit().aicc)
               except:
                   aicc.append(
                      sm.tsa.UnobservedComponents(
                      np.asarray(df[sku]),
                      exog = exog_fit,
                      **params,
                      freq_seasonal=[{'period':12,'harmonics':12}]).fit(method='cg').aicc)
        
        minaicc = UCM_all_params[np.argmin(aicc)]

        fitUCM = sm.tsa.UnobservedComponents(
            np.asarray(df[sku]),
            exog = exog_fit,
            **minaicc,
            freq_seasonal=[{'period':12,'harmonics':12}]).fit()
        df_UCM[sku] = fitUCM.forecast(fcperiod,exog = exog_fc)
        df_param.at[df_param[df_param['Model'] == 'UCM'].index[0],sku] = ( minaicc )
        #df_UCM.set_index(future_index,inplace=True)
        df_rmse.at[df_rmse[df_rmse['Model'] == 'UCM'].index[0],sku] = mean_squared_error(np.asarray(df[sku]), fitUCM.fittedvalues, squared=False)
    except:
        pass
    
    if df_rmse['Model'][np.argmin(df_rmse[sku])] == 'Prophet':
        df_best[sku] = df_P[sku]
    elif df_rmse['Model'][np.argmin(df_rmse[sku])] == 'SES':
        df_best[sku] = df_SES[sku]
    elif df_rmse['Model'][np.argmin(df_rmse[sku])] == 'Holt-Winter':
        df_best[sku] = df_HW[sku]
    elif df_rmse['Model'][np.argmin(df_rmse[sku])] == 'SARIMAX':
        df_best[sku] = df_SARIMA[sku]
    else:
        df_best[sku] = df_UCM[sku]


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




pd.concat([df_SES,df_P,df_HW,df_UCM,df_SARIMA],axis=0).to_excel(r'C:\Users\{}\Downloads\StatFC_Full.xlsx'.format(username))
df_param.to_excel(r'C:\Users\{}\Downloads\StatFC_Params.xlsx'.format(username))
df_rmse.to_excel(r'C:\Users\{}\Downloads\StatFC_RMSE.xlsx'.format(username))
df_best.to_excel(r'C:\Users\{}\Downloads\StatFC.xlsx'.format(username))




#plot

#fig, ax = plt.subplots()
#ax.plot(df[sku], color='k', label='data')
#ax.plot(df_UCM[sku], label='predicted')



