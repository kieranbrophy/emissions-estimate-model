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
    all_df[['economic_sector']] = all_df.economic_sector.astype('category')
    all_df[['iea_sector']] = all_df.iea_sector.astype('category')
    all_df[['region']] = all_df.region.astype('category')
    all_df[['iso2']] = all_df.iso2.astype('category')
    
    all_df['ind_code'] = all_df.industry.cat.codes
    all_df['eco_code']= all_df.economic_sector.cat.codes
    all_df['iea_code']= all_df.iea_sector.cat.codes
    all_df['reg_code'] = all_df.region.cat.codes
    all_df['iso_code'] = all_df.iso2.cat.codes
    
    train_df = all_df.dropna(subset = ['em_true'])
    
    X = train_df[['iea_code','ind_code','reg_code','iso_code','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    y = train_df[['iea_code','em_true']]
    
    X_train, X_test_tot, y_train, y_test_tot = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y['iea_code'])
    
    X_val, X_test, y_val, y_test = train_test_split(X_test_tot, y_test_tot, test_size=0.5, random_state=0, stratify=y_test_tot['iea_code'])
        
    X_real = all_df[['ind_code','eco_code','iea_code','reg_code','iso_code','va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']]
    
    BigBoosty = X_train.groupby('iea_code').apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['iea_code'] == x['iea_code'].iloc[0]],
                                                                                 X_test.loc[X_test['iea_code'] == x['iea_code'].iloc[0]], y_test.loc[y_test['iea_code'] == x['iea_code'].iloc[0]],
                                                                                 X_val.loc[X_test['iea_code'] == x['iea_code'].iloc[0]], y_val.loc[y_test['iea_code'] == x['iea_code'].iloc[0]],
                                                                                 X_real.loc[X_real['iea_code'] == x['iea_code'].iloc[0]]))
    BigBoosty = BigBoosty.reset_index('iea_code', drop=True)
    BigBoosty[PrimaryKey.assetid] = BigBoosty.index
        
    #boostyTest = eem.xgBoostyBoost(X_test, y_test, X_val, y_val, X_real)
    
    result = BigBoosty#[[PrimaryKey.assetid,'em_true','em_est']]

    
    return result
