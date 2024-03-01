# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:05:52 2023

@author: phu.pg
"""

from sklearn.metrics import mean_squared_error as MSE
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import datetime
from skopt import gp_minimize
from google.cloud import bigquery
from dateutil.relativedelta import relativedelta
#import os
#PROJECT 

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/OneDrive - pnj.com.vn/Docker/SA/service-account.json"
project_id = 'pnj-sc-aa-enhance'
client = bigquery.Client(project=project_id,location='asia-southeast1')

#NGAY DAU TIEN CUA TODAY()
fdcm = datetime.datetime.combine(datetime.date.today().replace(day=1), datetime.time.min)
#NGAY DAU TIEN CUA THANG TRUOC
lastmonth = fdcm + relativedelta(months=-1, day=1)

#DO DAI CUA DU BAO
fclength = 13

#make dataframe for forecast
def make_future_dataframe(self, periods, freq='MS'): #self la df sales history, periods la forecast length
    last_date = self.index.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,
        freq=freq)
    dates = dates[dates > last_date]
    dates = dates[:periods]
    return pd.DataFrame(index= dates)

#ACTUAL SALE
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

#PARAMETER CUA CAC LAN CHAY GAN NHAT
ucm_param_init = client.query(
"""
  SELECT DATE,REGION,PFSAP,PARAMETER,VALUE
  FROM
    `pnj-sc-aa-enhance.INPUT_FORECAST.MODEL_PARAMETER`
  WHERE MODEL = 'UCM' AND DATE = '{}'
""".format(lastmonth.strftime("%Y-%m-%d"))
).to_dataframe()
#CHUYEN DF TU FORMAT LONG SANG WIDE
ucm_param_init = ucm_param_init.pivot(index=['DATE','REGION','PFSAP'],columns='PARAMETER',values='VALUE').reset_index()
#CONVERT COLUMN DATE TU DANG STRING SANG DATE
ucm_param_init['DATE'] = pd.to_datetime(ucm_param_init['DATE'],format='%m%Y')
#FILTER LAY PARAMETER CUA LAN CHAY GAN NHAT
#ucm_param_init = ucm_param_init[ucm_param_init['DATE'] == lastmonth]
#TAO COT LOOKUP DE FILTER KHI CHAY VONG LAP
ucm_param_init['REGION-PFSAP'] = ucm_param_init['REGION'] + '|' + ucm_param_init['PFSAP']


##############################CLEAN DATA#############################
df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)
df = df[df['YEAR_MONTH'] >= '2020-01-01']
df = df[df['YEAR_MONTH'] < fdcm]
df = df.replace(0, np.nan)
df = df.dropna()
df['REGION-PFSAP'] = df['REGION']+'|'+df['PFSAP']
df = df.groupby(['YEAR_MONTH','REGION-PFSAP'])['QNT_ADJ'].sum().reset_index()
df = df.pivot(index='YEAR_MONTH', columns='REGION-PFSAP', values='QNT_ADJ')
df.index.name = 'Date'
df.index.freq = 'MS'
df = df.fillna(df.median())
df_UCM = make_future_dataframe(df,periods=fclength)
df_param = pd.DataFrame()


##############################Define search space for the model parameters#############################
space = [
    (0.1, 0.6),  # alpha_trend
    (0.1, 0.6),  # beta_trend
    (0.1, 0.6),  # phi_seasonal
    (0.1, 0.6),  # gamma_seasonal
    (True, False),  # trend
    (True, False),  # damped_cycle
    (True, False),  # irregular
    ['ntrend', 'dconstant', 'llevel', 'rwalk', 'dtrend', 'lldtrend', 'rwdrift', 'lltrend', 'strend', 'rtrend'],  # level_component
    (0.1, 0.6),  # level_scale
    (4,16), #harmonics
]
space_name = ['alpha','beta','phi','gamma','trend','damped_cycle','irregular','level','level_scale','harmonic','mse']
##############################HAM MUC TIEU#############################
def objective(params):
    alpha_trend, beta_trend, phi_seasonal, gamma_seasonal, trend, damped_cycle, \
        irregular, level_component, level_scale, harmonics = params
    model = sm.tsa.UnobservedComponents(
        df[sku].astype('float64'),
        level=level_component,
        trend=trend,
        freq_seasonal=[{'period':12,'harmonics':harmonics}],
        cycle=True,
        damped_cycle=damped_cycle,
        irregular=irregular,
        alpha=alpha_trend,
        beta=beta_trend,
        phi=phi_seasonal,
        gamma=gamma_seasonal,
        level_scale=level_scale
    )
    fitted_model = model.fit()
    y_pred = fitted_model.predict()
    mse = MSE(df[sku], y_pred)
    return mse

#############################CHAY DU BAO#############################    
for sku in list(df):
    #Bayesian optimization
    try:
        x0=ucm_param_init[ucm_param_init['REGION-PFSAP']==sku][['alpha','beta','phi','gamma','trend','damped_cycle','irregular','level','level_scale','harmonic']].values.tolist()
        result = gp_minimize(objective, space, n_calls=20,acq_func='EI',acq_optimizer='lbfgs',x0=x0,random_state=0,n_jobs=-1)
    except:
        result = gp_minimize(objective, space, n_calls=30,acq_func='gp_hedge',acq_optimizer='lbfgs',random_state=0,n_jobs=-1)
    model = sm.tsa.UnobservedComponents(
                                        df[sku].astype('float64'),
                                        level=result.x[7],
                                        trend=result.x[4],
                                        freq_seasonal=[{'period':12,'harmonics':result.x[9]}],
                                        cycle=True,
                                        irregular=result.x[6],
                                        damped_cycle=result.x[5],
                                        alpha=result.x[0],
                                        beta=result.x[1],
                                        phi=result.x[2],
                                        gamma=result.x[3],
                                        level_scale=result.x[8]).fit()
    df_UCM[sku] = model.forecast(fclength)
    result.x.append(result.fun)
    df_param[sku] = result.x
df_param['PARAMETER'] = space_name

##############################DAFRAME CHUA KET QUA DU BAO#############################
df_UCM.reset_index(inplace=True)
df_UCM = df_UCM.melt(id_vars='index')
df_UCM[['REGION', 'PFSAP']] = df_UCM['variable'].str.split('|', expand=True)
df_UCM.rename(columns={'index':'MONTH',
                       'value': 'QNT'
                       },inplace=True)
df_UCM['MONTH'] = pd.to_datetime(df_UCM['MONTH'])
df_UCM['MONTH'] = df_UCM['MONTH'].dt.date
df_UCM['MKDATE'] = fdcm.date()
df_UCM['MODEL'] = 'UCM'

##############################DAFRAME CHUA CAC PARAMETER#############################
df_param = df_param.melt(id_vars='PARAMETER')
df_param[['REGION', 'PFSAP']] = df_param['variable'].str.split('|', expand=True)
df_param['DATE'] = fdcm.date()
df_param['MODEL'] = 'UCM'
df_param.rename(columns={'value':'VALUE'},inplace=True)

#############################INSERT ROW TO FCT_MODEL#############################
FCT_MODEL = client.dataset('OUTPUT_FORECAST').table('FCT_MODEL')
'''
client.insert_rows_json(client.get_table(FCT_MODEL), 
                        df_UCM[['MONTH','MODEL','QNT','REGION','PFSAP','MKDATE']].to_dict('records')
                        )
client.insert_rows_from_dataframe(client.get_table(FCT_MODEL), 
                        df_UCM[['MONTH', 'MODEL', 'QNT', 'REGION', 'PFSAP', 'MKDATE']].astype(
                            {'MODEL': 'string', 'QNT': 'int64', 'REGION': 'string', 'PFSAP': 'string'})
                        )
'''
client.insert_rows(client.get_table(FCT_MODEL), 
                   df_UCM[['MONTH', 'MODEL', 'QNT', 'REGION', 'PFSAP', 'MKDATE']].astype(
                       {'MODEL': 'string', 'QNT': 'int64', 'REGION': 'string', 'PFSAP': 'string'}).to_dict('records')
                   )
##############################INSERT ROW TO MODEL_PARAMETER#############################
MODEL_PARAMETER = client.dataset('INPUT_FORECAST').table('MODEL_PARAMETER')
client.insert_rows(client.get_table(MODEL_PARAMETER), 
                   df_param[['DATE','REGION','PFSAP','PARAMETER','VALUE','MODEL']].astype(
                       {'REGION': 'string', 'PFSAP': 'string', 'PARAMETER': 'string', 'VALUE': 'string', 'MODEL': 'string'}).to_dict('records')
                   )
