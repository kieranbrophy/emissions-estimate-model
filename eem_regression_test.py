#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:12:51 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import pandas as pd
import eem_cal_functions as eem
import _config_ as config

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
    
        globals()[f"reg_global_{var}"] = train_df.groupby('industry').apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))
        globals()[f"reg_region_{var}"] = train_df.groupby(['industry','region']).apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))
        globals()[f"reg_country_{var}"] = train_df.groupby(['industry','iso2']).apply(lambda x: eem.linReg(x['em_true'], x[str(var)]))

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
        globals()[f"cal_global_{var}"] = globals()[f"reg_global_{var}"].merge(test_df_global, left_index=True, right_index=True)
        globals()[f"result_global_{var}"] = globals()[f"cal_global_{var}"].groupby('industry').apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept))
    
        globals()[f"cal_global_{var}"] = globals()[f"cal_global_{var}"].reset_index(0).reset_index(drop=True)
        globals()[f"result_global_{var}"] = globals()[f"result_global_{var}"].reset_index(0).reset_index(drop=True)
    
        globals()[f"result_global_{var}"]['em_est'] = abs(globals()[f"result_global_{var}"][0])
    
        globals()[f"result_global_comb_{var}"] = globals()[f"result_global_{var}"].merge(globals()[f"cal_global_{var}"], left_index=True, right_index=True, how='outer')
        globals()[f"result_global_comb_{var}"]['spatial_scale'] = 'Global'
        
        globals()[f"result_global_clean_{var}"] = globals()[f"result_global_comb_{var}"][['asset_id','em_est','em_true','r_sq','datapoints','std_err','p','industry_x','spatial_scale']]
        globals()[f"result_global_clean_{var}"].rename(columns={"industry_x": "industry"}) 
    
        '''
        Next work out emissions estimates on a continental scale
        ''' 
        globals()[f"cal_region_{var}"] = globals()[f"reg_region_{var}"].merge(test_df_region, left_index=True, right_index=True)
        globals()[f"result_region_{var}"] = globals()[f"cal_region_{var}"].groupby(['industry','region']).apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept))
    
        globals()[f"cal_region_{var}"] = globals()[f"cal_region_{var}"].reset_index(0).reset_index(drop=True)
        globals()[f"result_region_{var}"] = globals()[f"result_region_{var}"].reset_index(0).reset_index(drop=True)
    
        globals()[f"result_region_{var}"]['em_est'] = abs(globals()[f"result_region_{var}"][0])
    
        globals()[f"result_region_comb_{var}"] = globals()[f"result_region_{var}"].merge(globals()[f"cal_region_{var}"], left_index=True, right_index=True, how='outer')
        globals()[f"result_region_comb_{var}"]['spatial_scale'] = 'Regional'
        
        globals()[f"result_region_clean_{var}"] = globals()[f"result_region_comb_{var}"][['asset_id','em_est','em_true','r_sq','datapoints','std_err','p','industry_x','spatial_scale']]
        globals()[f"result_region_clean_{var}"].rename(columns={"industry_x": "industry"})
        
        '''
        Finally work out emissions estimates on a country scale
        '''
        globals()[f"cal_country_{var}"] = globals()[f"reg_country_{var}"].merge(test_df_country, left_index=True, right_index=True)
        globals()[f"result_country_{var}"] = globals()[f"cal_country_{var}"].groupby(['industry','iso2']).apply(lambda x: eem.calcem(x[str(var)],x.slope,x.intercept))
    
        globals()[f"cal_country_{var}"] = globals()[f"cal_country_{var}"].reset_index(0).reset_index(drop=True)
        globals()[f"result_country_{var}"] = globals()[f"result_country_{var}"].reset_index(0).reset_index(drop=True)
    
        globals()[f"result_country_{var}"]['em_est'] = abs(globals()[f"result_country_{var}"][0])
    
        globals()[f"result_country_comb_{var}"] = globals()[f"result_country_{var}"].merge(globals()[f"cal_country_{var}"], left_index=True, right_index=True, how='outer')
        globals()[f"result_country_comb_{var}"]['spatial_scale'] = 'Country'
        
        globals()[f"result_country_clean_{var}"] = globals()[f"result_country_comb_{var}"][['asset_id','em_est','em_true','r_sq','datapoints','std_err','p','industry_x','spatial_scale']]
        globals()[f"result_country_clean_{var}"].rename(columns={"industry_x": "industry"})
    
        '''
        Now we build the optimal emission estimate for each test entity, starting with the global result
        '''
        globals()[f"result_{var}"] = globals()[f"result_global_clean_{var}"]
        
        '''
        Merge optimised results together based on geography
        '''
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_region_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_region_clean_{var}"]['p']) & (globals()[f"result_region_clean_{var}"]['datapoints'] > config.min_datapoints), 'em_est'] = globals()[f"result_region_clean_{var}"]['em_est']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_region_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_region_clean_{var}"]['p']) & (globals()[f"result_region_clean_{var}"]['datapoints'] > config.min_datapoints), 'r_sq'] = globals()[f"result_region_clean_{var}"]['r_sq']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_region_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_region_clean_{var}"]['p']) & (globals()[f"result_region_clean_{var}"]['datapoints'] > config.min_datapoints), 'p'] = globals()[f"result_region_clean_{var}"]['p']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_region_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_region_clean_{var}"]['p']) & (globals()[f"result_region_clean_{var}"]['datapoints'] > config.min_datapoints), 'datapoints'] = globals()[f"result_region_clean_{var}"]['datapoints']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_region_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_region_clean_{var}"]['p']) & (globals()[f"result_region_clean_{var}"]['datapoints'] > config.min_datapoints), 'spatial_scale'] = globals()[f"result_region_clean_{var}"]['spatial_scale']
        
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_country_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_country_clean_{var}"]['p']) & (globals()[f"result_country_clean_{var}"]['datapoints'] > config.min_datapoints), 'em_est'] = globals()[f"result_country_clean_{var}"]['em_est']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_country_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_country_clean_{var}"]['p']) & (globals()[f"result_country_clean_{var}"]['datapoints'] > config.min_datapoints), 'r_sq'] = globals()[f"result_country_clean_{var}"]['r_sq']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_country_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_country_clean_{var}"]['p']) & (globals()[f"result_country_clean_{var}"]['datapoints'] > config.min_datapoints), 'p'] = globals()[f"result_region_country_{var}"]['p']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_country_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_country_clean_{var}"]['p']) & (globals()[f"result_country_clean_{var}"]['datapoints'] > config.min_datapoints), 'datapoints'] = globals()[f"result_country_clean_{var}"]['datapoints']
        globals()[f"result_{var}"].loc[(globals()[f"result_{var}"]['r_sq'] < globals()[f"result_country_clean_{var}"]['r_sq']) & (globals()[f"result_{var}"]['p'] > globals()[f"result_country_clean_{var}"]['p']) & (globals()[f"result_country_clean_{var}"]['datapoints'] > config.min_datapoints), 'spatial_scale'] = globals()[f"result_country_clean_{var}"]['spatial_scale']
        
                
    result = result_va_usd
    result['variable'] = 'va_usd'
       
    for var in variables:
        '''
        Merge optimised results together based on variable
        '''
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'em_est'] = globals()[f"result_{var}"]['em_est']
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'r_sq'] = globals()[f"result_{var}"]['r_sq']
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'p'] = globals()[f"result_{var}"]['p']
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'datapoints'] = globals()[f"result_{var}"]['datapoints']
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'spatial_scale'] = globals()[f"result_{var}"]['spatial_scale']
        result.loc[(result['r_sq'] < globals()[f"result_{var}"]['r_sq']) & (result['p'] > globals()[f"result_{var}"]['p']) & (globals()[f"result_{var}"]['datapoints'] > config.min_datapoints), 'variable'] = str(var)
        

    return result

'''
Compute multi variable regression using all independent variables
'''
def multiReg(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Set approporate multi index for global, regional and country
    '''
    test_df_global = test_df.set_index(['industry'])
    test_df_region = test_df.set_index(['industry','region'])
    test_df_country = test_df.set_index(['industry','iso2'])
    
    '''
    Compute regression globally, regionally and by country
    '''
    reg_global = train_df.groupby('industry').apply(lambda x: eem.multiLinReg(x[config.variables], x['em_true'],test_df_global[config.variables]))
    reg_region = train_df.groupby(['industry','region']).apply(lambda x: eem.multiLinReg(x[config.variables], x['em_true'],test_df_region[config.variables]))
    reg_country = train_df.groupby(['industry','iso2']).apply(lambda x: eem.multiLinReg(x[config.variables], x['em_true'],test_df_country[config.variables]))
    
    '''
    First work out emissions estimates on a global scale
    '''
    cal_global = reg_global.merge(test_df_global, left_index=True, right_index=True)
    result_global = cal_global.groupby('asset_id').apply(lambda x: eem.calcem_multi(x[config.variables], x['slope'], x['intercept']))
        
    cal_global = cal_global.reset_index(0).reset_index(drop=True)
    result_global = result_global.reset_index(0).reset_index(drop=True)
        
    result_global[['em_est']] = result_global[0].str.get(0)
        
    result_global_comb = result_global.merge(cal_global, left_index=True, right_index=True, how='outer')
    result_global_clean = result_global_comb[['asset_id_x','industry','em_est','em_true','r_sq','datapoints']]
    result_global_clean = result_global_clean.rename(columns={"asset_id_x": "asset_id"}) 
        
    '''
    Next work out emissions estimates on a continental scale
    ''' 
    
    cal_region = reg_region.merge(test_df_region, left_index=True, right_index=True)
    result_region = cal_region.groupby('asset_id').apply(lambda x: eem.calcem_multi(x[config.variables], x['slope'], x['intercept']))
        
    cal_region = cal_region.reset_index(0).reset_index(drop=True)
    result_region = result_region.reset_index(0).reset_index(drop=True)
        
    result_region[['em_est']] = result_region[0].str.get(0)
        
    result_region_comb = result_region.merge(cal_global, left_index=True, right_index=True, how='outer')
    result_region_clean = result_region_comb[['asset_id_x','industry','em_est','em_true','r_sq','datapoints']]
    result_region_clean = result_region_clean.rename(columns={"asset_id_x": "asset_id"}) 
            
    '''
    #Finally work out emissions estimates on a country scale
    '''
    cal_country = reg_country.merge(test_df_country, left_index=True, right_index=True)
    result_country = cal_country.groupby('asset_id').apply(lambda x: eem.calcem_multi(x[config.variables], x['slope'], x['intercept']))
        
    cal_country = cal_country.reset_index(0).reset_index(drop=True)
    result_country = result_country.reset_index(0).reset_index(drop=True)
        
    result_country[['em_est']] = result_country[0].str.get(0)
        
    result_country_comb = result_country.merge(cal_global, left_index=True, right_index=True, how='outer')
    result_country_clean = result_country_comb[['asset_id_x','industry','em_est','em_true','r_sq','datapoints']]
    result_country_clean = result_country_clean.rename(columns={"asset_id_x": "asset_id"}) 
    
    '''
    Merge optimised results together based on geography
    '''
    result = result_global_clean
    
    result.loc[(result['r_sq'] < result_region_clean['r_sq']) & (result_region_clean['datapoints'] > min_datapoints), 'em_est'] = result_region_clean['em_est']
    result.loc[(result['r_sq'] < result_region_clean['r_sq']) & (result_region_clean['datapoints'] > min_datapoints), 'r_sq'] = result_region_clean['r_sq']
    result.loc[(result['r_sq'] < result_region_clean['r_sq']) & (result_region_clean['datapoints'] > min_datapoints), 'datapoints'] = result_region_clean['datapoints']
    
    result.loc[(result['r_sq'] < result_country_clean['r_sq']) & (result_country_clean['datapoints'] > min_datapoints), 'em_est'] = result_country_clean['em_est']
    result.loc[(result['r_sq'] < result_country_clean['r_sq']) & (result_country_clean['datapoints'] > min_datapoints), 'r_sq'] = result_country_clean['r_sq']
    result.loc[(result['r_sq'] < result_country_clean['r_sq']) & (result_country_clean['datapoints'] > min_datapoints), 'datapoints'] = result_country_clean['datapoints']
    
    return result