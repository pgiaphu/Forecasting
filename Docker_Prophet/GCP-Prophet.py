# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:32:52 2024

@author: phu.pg
"""
import optuna
from prophet import Prophet
import pandas as pd 
import numpy as np
import datetime
from google.cloud import bigquery
from dateutil.relativedelta import relativedelta
#import os
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as MASE
#PROJECT 

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/OneDrive - pnj.com.vn/SA/service-account-enhance.json"
project_id = 'pnj-sc-aa-enhance'
client = bigquery.Client(project=project_id,location='asia-southeast1')

#NGAY DAU TIEN CUA TODAY()
fdcm = datetime.datetime.combine(datetime.date.today().replace(day=1), datetime.time.min)

#NGAY DAU TIEN CUA THANG TRUOC
lastmonth = fdcm + relativedelta(months=-1, day=1)
lag3 = fdcm + relativedelta(months=-3, day=1)
fclength = 13


##########################HOLIDAY###################################
hld = client.query(
'''
  SELECT DISTINCT
    FIRST_DATE_OF_MONTH as ds,
    HOLIDAY_NAME as holiday,
  FROM
    `pnj-sc-aa-enhance.GENERAL.W_DATE_D`
  WHERE
    FIRST_DATE_OF_MONTH >= "2020-01-01"
''').to_dataframe()
hld = hld.assign(holiday=hld['holiday'].str.split(' và ')).explode('holiday').reset_index(drop=True)
hld = hld[['ds','holiday']].drop_duplicates().sort_values('ds')
hld = hld[~hld.ds.isin(['2020-04-01','2021-07-01', '2021-08-01', '2021-09-01'])]
covid = pd.DataFrame({
  'holiday': 'COVID19',
  'ds': pd.to_datetime(['2020-04-01','2021-07-01', '2021-08-01', '2021-09-01'
                        ]),
})
hld=pd.concat([hld,covid])


##############################PARAMETER CUA LAN CHAY GAN NHAT##########################################
prophet_param_init = client.query(
"""
  SELECT REGION,PFSAP,PARAMETER,VALUE
  FROM
    `pnj-sc-aa-enhance.INPUT_FORECAST.MODEL_PARAMETER`
  WHERE MODEL = 'Prophet' AND DATE = '{}'
""".format(lastmonth.strftime("%Y-%m-%d"))
).to_dataframe()
#CHUYEN DF TU FORMAT LONG SANG WIDE
prophet_param_init = prophet_param_init.pivot(index=['REGION','PFSAP'],columns='PARAMETER',values='VALUE').reset_index()
#TAO COT LOOKUP DE FILTER KHI CHAY VONG LAP
prophet_param_init['REGION-PFSAP'] = prophet_param_init['REGION'] + '|' + prophet_param_init['PFSAP']

###############################ACTUAL SALE##############################
df = client.query(
'''
  SELECT
    YEAR_MONTH,
    REGION_CODE,
    REGION,
    PFSAP,
    NET_QTY AS QNT,
    ADJ_QTY AS QNT_ADJ
  FROM
    `pnj-sc-aa-enhance.INPUT_FORECAST.W_INPUT_FC_DATABYMONTH_AGG`
  WHERE
    PFSAP NOT IN ('None')  
''').to_dataframe().set_index('YEAR_MONTH')

df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)
df = df[df['YEAR_MONTH'] >= '2020-01-01']
df = df[df['YEAR_MONTH'] < fdcm]
df = df.replace(0, np.nan)
df = df.dropna()
df['REGION-PFSAP'] = df['REGION']+'|'+df['PFSAP']
df = df.groupby(['YEAR_MONTH','REGION-PFSAP'])['QNT_ADJ'].sum().reset_index()
df = df.rename(columns={'YEAR_MONTH':'ds','QNT_ADJ':'y'})
df['y'] = df['y'].astype(int)
df = df.loc[df['ds'] < fdcm]
#bo cac PFSAP-REGION khong xuat ban trong 3 thang gan nhat
df = df[~df['REGION-PFSAP'].isin(
                                df.groupby('REGION-PFSAP')['ds'].max().loc[lambda x: x < lag3].index.unique().tolist()
                                )]

df_final = pd.DataFrame()
Prophet_param = pd.DataFrame()
###########################################OPTUNA FUCNTION#####################################
def objective(trial):
    # Define the parameter search space for hyperparameter tuning
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.2),
        'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.9),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 0.5),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0),
        'growth': trial.suggest_categorical('growth', ['linear', 'logistic','flat']),
    }
    
    m = Prophet(**params,
                uncertainty_samples=0,
                interval_width=None,
                mcmc_samples=0,
                weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=True,
                holidays=hld
                    )
    #m.add_seasonality(name='monthly', period=12, fourier_order=12)
    m.add_country_holidays(country_name='VN')
    m.fit(train_data[['ds','y','cap','floor']])
    preds = m.predict(test_data[['ds', 'cap', 'floor']])
    
    mae_score = MASE(test_data['y'], preds['yhat'],y_train=train_data['y'],sp=12) 
    #MAPE(test_data['y'], preds['yhat'])
    #MASE(test_data['y'], preds['yhat'],y_train=train_data['y'])       
    #MSE(test_data['y'], preds['yhat'],squared=False)
    
    return mae_score

#####################################FORECAST SKU###################################
for sku in df['REGION-PFSAP'].unique():
    df_model = df[df['REGION-PFSAP'] == sku]
    df_model['cap'] = df_model.y.quantile(0.95)+1 # UCL
    df_model['floor'] = df_model.y.quantile(0.1) # LCL

    train_size = int(0.85 * len(df_model))
    train_data = df_model[:train_size]
    test_data = df_model[train_size:]

    study = optuna.create_study(direction='minimize')
    try:
        #SEARCH XUNG QUANH GIA TRI TOT NHAT CUA LAN CHAY TRUOC DO
        study.enqueue_trial({key: float(value) if i in [0,1,3,4] else value for i, (key, value) in enumerate(prophet_param_init[prophet_param_init['REGION-PFSAP'] == sku].iloc[:,2:-1].to_dict(orient='records')[0].items())})
    except:
        pass
    try:
        study.optimize(objective, n_trials=30,n_jobs=-1)
    except:
        continue
    


    m = Prophet(**study.best_params,
                uncertainty_samples=0,
                interval_width=None,
                mcmc_samples=0,
                weekly_seasonality=False,daily_seasonality=False,yearly_seasonality=True,
                holidays=hld
                    )
    #m.add_seasonality(name='monthly', period=12, fourier_order=12)
    m.add_country_holidays(country_name='VN')
    m.fit(df_model[['ds','y','cap','floor']])
    preds = m.predict(df_model[['ds', 'cap', 'floor']])
    
    future = m.make_future_dataframe(periods=fclength,freq='MS',include_history=False)            
    future['cap'] = df_model.y.quantile(0.95)+1
    future['floor'] = df_model.y.quantile(0.1)
    forecast = m.predict(future)[['ds','yhat']]
    forecast['REGION-PFSAP'] = sku
    
    df_final = pd.concat([df_final,forecast])
    Prophet_param = pd.concat([Prophet_param,
                               pd.DataFrame([study.best_params]).assign(**{'REGION-PFSAP': sku})
                               ])
##############################DAFRAME CHUA KET QUA DU BAO#############################
df_final[['REGION', 'PFSAP']] = df_final['REGION-PFSAP'].str.split('|', expand=True)
df_final['MKDATE'] = fdcm.date()
df_final['MODEL'] = 'Prophet'
df_final.rename(columns={'ds':'MONTH',
                       'yhat': 'QNT',                     
                       },inplace=True)
df_final = df_final[df_final['QNT']>0]
##############################DAFRAME CHUA CAC PARAMETER#############################
Prophet_param = Prophet_param.melt(id_vars='REGION-PFSAP')
Prophet_param[['REGION', 'PFSAP']] = Prophet_param['REGION-PFSAP'].str.split('|', expand=True)
Prophet_param['DATE'] = fdcm.date()
Prophet_param['MODEL'] = 'Prophet'
Prophet_param.rename(columns={'value':'VALUE',
                              'variable':'PARAMETER'
                              },inplace=True)


#############################INSERT ROW TO FCT_MODEL#############################
FCT_MODEL = client.dataset('OUTPUT_FORECAST').table('FCT_MODEL')
client.insert_rows(client.get_table(FCT_MODEL), 
                   df_final[['MONTH', 'MODEL', 'QNT', 'REGION', 'PFSAP', 'MKDATE']].astype(
                       {'MODEL': 'string', 'QNT': 'int64', 'REGION': 'string', 'PFSAP': 'string'}).to_dict('records')
                   )
##############################INSERT ROW TO MODEL_PARAMETER#############################
MODEL_PARAMETER = client.dataset('INPUT_FORECAST').table('MODEL_PARAMETER')
client.insert_rows(client.get_table(MODEL_PARAMETER), 
                   Prophet_param[['DATE','REGION','PFSAP','PARAMETER','VALUE','MODEL']].astype(
                       {'REGION': 'string', 'PFSAP': 'string', 'PARAMETER': 'string', 'VALUE': 'string', 'MODEL': 'string'}).to_dict('records')
                   )


