#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:12:51 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

import eem_cal_functions as eem
import _config_ as config
from sray_db.apps.pk import PrimaryKey

min_datapoints = config.min_datapoints
variables = config.variables
    

'''
Compute single variable regression for each independent variable
'''
def singleReg(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute regression for each variable globally, regionally and by country
    '''
    for var in variables:
    
        globals()[f"reg_global_{var}"] = train_df.dropna(subset = [str(var)]).groupby('industry').apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))
        globals()[f"reg_region_{var}"] = train_df.dropna(subset = [str(var)]).groupby(['industry','region']).apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))
        globals()[f"reg_country_{var}"] = train_df.dropna(subset = [str(var)]).groupby(['industry','iso2']).apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))

    '''
    Set approporate multi index for global, regional and country
    '''
    test_df_global = test_df.set_index(['industry'])
    test_df_region = test_df.set_index(['industry','region'])
    test_df_country = test_df.set_index(['industry','iso2'])

    '''
    Calculate emissions estimate per FactSet sector for each variable using the results of the single linear regression
    '''
    for var in variables:
    
        '''
        First work out emissions estimates on a global scale
        '''
        globals()[f"cal_global_{var}"] = globals()[f"reg_global_{var}"].merge(test_df_global, left_index=True, right_index=True).reset_index()      
        
        globals()[f"result_global_{var}"] = globals()[f"cal_global_{var}"].groupby([PrimaryKey.assetid]).apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept)).to_frame()
        globals()[f"result_global_{var}"]['em_est'] = abs(globals()[f"result_global_{var}"].iloc[:, 0])
        
        globals()[f"result_global_comb_{var}"] = globals()[f"result_global_{var}"].merge(globals()[f"cal_global_{var}"], on=[PrimaryKey.assetid])
        globals()[f"result_global_comb_{var}"]['spatial_scale'] = 'Global'
        
        globals()[f"result_global_clean_{var}"] = globals()[f"result_global_comb_{var}"][[PrimaryKey.assetid,'em_est','em_true','r_sq','datapoints','std_err','p','industry','spatial_scale']]
        globals()[f"result_global_clean_{var}"].set_index([PrimaryKey.assetid])
    
        '''
        Next work out emissions estimates on a continental scale
        ''' 
        globals()[f"cal_region_{var}"] = globals()[f"reg_region_{var}"].merge(test_df_region, left_index=True, right_index=True).reset_index()
        
        globals()[f"result_region_{var}"] = globals()[f"cal_region_{var}"].groupby([PrimaryKey.assetid]).apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept)).to_frame()    
        globals()[f"result_region_{var}"]['em_est'] = abs(globals()[f"result_region_{var}"].iloc[:, 0])
    
        globals()[f"result_region_comb_{var}"] = globals()[f"result_region_{var}"].merge(globals()[f"cal_region_{var}"], on=[PrimaryKey.assetid])
        globals()[f"result_region_comb_{var}"]['spatial_scale'] = 'Regional'

        globals()[f"result_region_clean_{var}"] = globals()[f"result_region_comb_{var}"][[PrimaryKey.assetid,'em_est','em_true','r_sq','datapoints','std_err','p','industry','spatial_scale']]
        globals()[f"result_region_clean_{var}"].set_index([PrimaryKey.assetid])
        
        '''
        Finally work out emissions estimates on a country scale
        '''
        globals()[f"cal_country_{var}"] = globals()[f"reg_country_{var}"].merge(test_df_country, left_index=True, right_index=True).reset_index()      
        
        globals()[f"result_country_{var}"] = globals()[f"cal_country_{var}"].groupby([PrimaryKey.assetid]).apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept)).to_frame()
        globals()[f"result_country_{var}"]['em_est'] = abs(globals()[f"result_country_{var}"].iloc[:, 0])
    
        globals()[f"result_country_comb_{var}"] = globals()[f"result_country_{var}"].merge(globals()[f"cal_country_{var}"], on=[PrimaryKey.assetid])
        globals()[f"result_country_comb_{var}"]['spatial_scale'] = 'Country'
        
        globals()[f"result_country_clean_{var}"] = globals()[f"result_country_comb_{var}"][[PrimaryKey.assetid,'em_est','em_true','r_sq','datapoints','std_err','p','industry','spatial_scale']]
        globals()[f"result_country_clean_{var}"].set_index(PrimaryKey.assetid)
    
        '''
        Now we build the optimal emission estimate for each test entity, starting with the global result
        '''
        globals()[f"result_{var}"] = globals()[f"result_global_clean_{var}"]
        
        '''
        Merge optimised results together based on geography
        '''
        for assetid in globals()[f"result_region_clean_{var}"].index:
            
            if globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['r_sq'][assetid] < globals()[f"result_region_clean_{var}"].loc[globals()[f"result_region_clean_{var}"].index == assetid]['r_sq'][assetid] and globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['p'][assetid] > globals()[f"result_region_clean_{var}"].loc[globals()[f"result_region_clean_{var}"].index == assetid]['p'][assetid] and globals()[f"result_region_clean_{var}"].loc[globals()[f"result_region_clean_{var}"].index == assetid]['datapoints'][assetid] > config.min_datapoints:
                
                globals()[f"result_{var}"].at[assetid,'em_est'] = globals()[f"result_region_clean_{var}"].at[assetid,'em_est']
                globals()[f"result_{var}"].at[assetid,'r_sq'] = globals()[f"result_region_clean_{var}"].at[assetid,'r_sq']
                globals()[f"result_{var}"].at[assetid,'p'] = globals()[f"result_region_clean_{var}"].at[assetid,'p']
                globals()[f"result_{var}"].at[assetid,'datapoints'] = globals()[f"result_region_clean_{var}"].at[assetid,'datapoints']
                globals()[f"result_{var}"].at[assetid,'spatial_scale'] = globals()[f"result_region_clean_{var}"].at[assetid,'spatial_scale']

        for assetid in globals()[f"result_country_clean_{var}"].index:
            
            if globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['r_sq'][assetid] < globals()[f"result_country_clean_{var}"].loc[globals()[f"result_country_clean_{var}"].index == assetid]['r_sq'][assetid] and globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['p'][assetid] > globals()[f"result_country_clean_{var}"].loc[globals()[f"result_country_clean_{var}"].index == assetid]['p'][assetid] and globals()[f"result_country_clean_{var}"].loc[globals()[f"result_country_clean_{var}"].index == assetid]['datapoints'][assetid] > config.min_datapoints:
            
                globals()[f"result_{var}"].at[assetid,'em_est'] = globals()[f"result_country_clean_{var}"].at[assetid,'em_est']
                globals()[f"result_{var}"].at[assetid,'r_sq'] = globals()[f"result_country_clean_{var}"].at[assetid,'r_sq']
                globals()[f"result_{var}"].at[assetid,'p'] = globals()[f"result_country_clean_{var}"].at[assetid,'p']
                globals()[f"result_{var}"].at[assetid,'datapoints'] = globals()[f"result_country_clean_{var}"].at[assetid,'datapoints']
                globals()[f"result_{var}"].at[assetid,'spatial_scale'] = globals()[f"result_country_clean_{var}"].at[assetid,'spatial_scale']    
                
    result = result_va_usd
    result['variable'] = 'va_usd'
       
    for var in variables:
        '''
        Merge optimised results together based on variable
        '''
        for assetid in globals()[f"result_{var}"].index:
            
            if result.loc[result.index == assetid]['r_sq'][assetid] < globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['r_sq'][assetid] and result.loc[result.index == assetid]['p'][assetid] > globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['p'][assetid] and globals()[f"result_{var}"].loc[globals()[f"result_{var}"].index == assetid]['datapoints'][assetid] > config.min_datapoints:
            
                result.at[assetid,'em_est'] = globals()[f"result_{var}"].at[assetid,'em_est']
                result.at[assetid,'r_sq'] = globals()[f"result_{var}"].at[assetid,'r_sq']
                result.at[assetid,'p'] = globals()[f"result_{var}"].at[assetid,'p']
                result.at[assetid,'datapoints'] = globals()[f"result_{var}"].at[assetid,'datapoints']
                result.at[assetid,'spatial_scale'] = globals()[f"result_{var}"].at[assetid,'spatial_scale']
                result.loc[assetid,'variable'] = str(var)

    result = result.dropna(subset = ['em_est'])
    
    return result

def xgBoost(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    
    X_test = test_df[[PrimaryKey.assetid,'industry','region','iso2', 'va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    y_test = test_df[[PrimaryKey.assetid,'industry','region','iso2','em_true']]
    
    X_train = train_df[[PrimaryKey.assetid,'industry','region','iso2', 'va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    y_train = train_df[[PrimaryKey.assetid,'industry','region','iso2','em_true']]

    boost_global = X_train.groupby('industry').apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['industry'] == x['industry'].iloc[0]], X_test.loc[X_test['industry'] == x['industry'].iloc[0]], y_test.loc[y_test['industry'] == x['industry'].iloc[0]]))
    boost_global = pd.DataFrame(boost_global)    
    boost_global['em_est'] = boost_global[0].str.get(0)

    result_global = y_test.merge(boost_global[['em_est','datapoints']], on = PrimaryKey.assetid, how='outer')
    result_global['spatial_level'] = 'global'

    boost_region = X_train.groupby(['industry','region']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['region'] == x['region'].iloc[0]], X_test.loc[X_test['region'] == x['region'].iloc[0]], y_test.loc[y_test['region'] == x['region'].iloc[0]]))
    boost_region = pd.DataFrame(boost_region) 
    boost_region['em_est'] = boost_region[0].str.get(0)
    
    result_region = y_test.merge(boost_region[['em_est','datapoints']], on = PrimaryKey.assetid, how='outer')
    result_region['spatial_level'] = 'regional'

    boost_country = X_train.groupby(['industry','iso2']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['iso2'] == x['iso2'].iloc[0]], X_test.loc[X_test['iso2'] == x['iso2'].iloc[0]], y_test.loc[y_test['iso2'] == x['iso2'].iloc[0]]))
    boost_country = pd.DataFrame(boost_country)    
    boost_country['em_est'] = boost_country[0].str.get(0)
    
    result_country = y_test.merge(boost_country[['em_est','datapoints']], on = PrimaryKey.assetid, how='outer')
    result_country['spatial_level'] = 'country'

    result_opt = result_country
    
    result_opt['em_est'].fillna(result_region['em_est'], inplace=True)
    result_opt['datapoints'].fillna(result_region['datapoints'], inplace=True)
    
    result_opt['em_est'].fillna(result_global['em_est'], inplace=True)
    result_opt['datapoints'].fillna(result_global['datapoints'], inplace=True)
    result_opt['spatial_level'].fillna(result_global['spatial_level'], inplace=True)

    result = result_opt

    
    return result
