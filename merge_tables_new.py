#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:37:28 2022

@author: kieran.brophyarabesque.com
"""
from datetime import date, datetime, timedelta
import pandas as pd

from sray_db.apps.pk import PrimaryKey

import _config_ as config

def mergeTables(info_df: pd.DataFrame, em_df: pd.DataFrame, tf_df: pd.DataFrame, mktcap_df: pd.DataFrame, employees_df: pd.DataFrame, oth_fin_df: pd.DataFrame)-> pd.DataFrame:
    
    
    data1_df = pd.merge_asof(oth_fin_df.sort_values(by=PrimaryKey.date),
                                tf_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta('2y'))

    data2_df = pd.merge_asof(mktcap_df.sort_values(by=PrimaryKey.date),
                                employees_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta('2y'))
    
    data_all_df = pd.merge_asof(data1_df.sort_values(by=PrimaryKey.date),
                                data2_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta('2y'))

    data_all_df = pd.merge_asof(data_all_df.sort_values(by=PrimaryKey.date),
                                em_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta('2y'))

    data_all_df = data_all_df.sort_values(by=[PrimaryKey.date], ascending=False)
    data_all_df = data_all_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')

    all_df = info_df.merge(data_all_df, on=PrimaryKey.assetid)
    all_df = all_df.dropna(subset=config.variables, thresh=2)
    all_df = all_df[[PrimaryKey.assetid, PrimaryKey.date,'industry','economic_sector','iea_sector','region','iso2','em_true', 'va_usd', 'revenue', 'employees', 'mktcap_avg_12m', 'ff_assets', 'ff_eq_tot', 'ff_mkt_val']] 
        
    return all_df