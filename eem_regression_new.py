#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:12:51 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import pandas as pd
from sklearn.model_selection import train_test_split

import eem_cal_functions_new as eem
import _config_ as config
from sray_db.apps.pk import PrimaryKey

def xgBoost(all_df) -> pd.DataFrame:
    
    all_df[['industry']] = all_df.industry.astype('category')
    all_df[['region']] = all_df.region.astype('category')
    all_df[['iso2']] = all_df.iso2.astype('category')
    
    all_df['ind_code'] = all_df.industry.cat.codes
    all_df['reg_code'] = all_df.region.cat.codes
    all_df['iso_code'] = all_df.iso2.cat.codes
    
    train_df = all_df.dropna(subset = ['em_true'])
    
    X = train_df[['ind_code','reg_code','iso_code','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    y = train_df[['em_true']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #X_val, X_test, y_val, y_test = train_test_split(X_test_tot, y_test_tot, test_size=0.5, random_state=42)
    
    X_real = all_df[['ind_code','reg_code','iso_code','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    
    BigBoosty = eem.xgBoostyBoost(X_train, y_train, X_test, y_test, X_real)
    
    #boostyTest = eem.xgBoostyBoost(X_test, y_test, X_val, y_val, X_real)
    
    boost = pd.DataFrame(BigBoosty) 
    print(boost)
    boost['em_est'] = boost[0].str.get(0)
    boost = boost.dropna(subset = ['em_est'])

    if config.test == True:
        result = y_test.merge(boost[['em_est','datapoints']], left_index=True, right_index=True)
    else:
        result = X_real.merge(boost[['em_est','datapoints']], left_index=True, right_index=True)
    
    return result
