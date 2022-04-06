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
from sray_db.apps.pk import PrimaryKey

import _config_ as config

def xgBoost(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    
    X_test = test_df[[PrimaryKey.assetid,'industry','region','iso2','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    y_test = test_df[[PrimaryKey.assetid,'industry','region','iso2','em_true']]
    
    X_train = train_df[[PrimaryKey.assetid,'industry','region','iso2','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
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
