#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:37:28 2022

@author: kieran.brophyarabesque.com
"""
import pandas as pd
import numpy as np

from functools import reduce

from sray_db.pk import PrimaryKey

import _config_ as config

def mergeTables(info_df, em_df, tf_df, employees_df, energy_df, ghg_df, oth_fin_df):
    
    employees_df = employees_df.sort_values(by=[PrimaryKey.date], ascending=False)
    employees_df = employees_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    energy_df = energy_df.sort_values(by=[PrimaryKey.date], ascending=False)
    energy_df = energy_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    ghg_df = ghg_df.sort_values(by=[PrimaryKey.date], ascending=False)
    ghg_df = ghg_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    oth_fin_df = oth_fin_df.sort_values(by=[PrimaryKey.date], ascending=False)
    oth_fin_df = oth_fin_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    merge_df = [info_df, em_df, tf_df, employees_df, energy_df, ghg_df, oth_fin_df]
    
    all_df = reduce(lambda  left,right: pd.merge(left,right,on=[PrimaryKey.assetid],
                                            how='outer'), merge_df)

    all_df = all_df.dropna(subset=config.variables_short, axis=0, thresh=config.nan_thresh)
    all_df = all_df.set_index([PrimaryKey.assetid])
    all_df = all_df[config.variables_long] 
    
    all_df = all_df.fillna(value=np.nan)
    all_df['ghg'] = pd.to_numeric(all_df['ghg'])
            
    return all_df

def oldEm(oldEm_df, oldtf_df, employees_df, scope):
    
    oldemp_df = employees_df

    oldEm_df = oldEm_df.sort_values(by=[PrimaryKey.date], ascending=False)
    oldEm_df = oldEm_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    mergeOld_df = pd.merge_asof(oldEm_df.sort_values(by=PrimaryKey.date),
                                oldtf_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta(14,'d'))
    
    mergeOld_df = mergeOld_df.sort_values(by=[PrimaryKey.date], ascending=False)
    mergeOld_df = mergeOld_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')

    allOld_df = pd.merge_asof(mergeOld_df.sort_values(by=PrimaryKey.date),
                                oldemp_df.sort_values(by=PrimaryKey.date),
                                on=[PrimaryKey.date],
                                by=[PrimaryKey.assetid],
                                tolerance=pd.Timedelta(365,'d'))

    
    allOld_df = allOld_df.sort_values(by=[PrimaryKey.date], ascending=False)
    allOld_df = allOld_df.drop_duplicates(subset=[PrimaryKey.assetid], keep='first')
    
    old_df = allOld_df.set_index([PrimaryKey.assetid])
    
    if scope == 'one':
        old_df = old_df.dropna(subset = ["em_1"]).rename(columns={"em_1": "em_true"})
    elif scope == 'two':
        old_df = old_df.dropna(subset = ["em_2"]).rename(columns={"em_2": "em_true"})
    elif scope == 'three':
        old_df = old_df.dropna(subset = ["em_3"]).rename(columns={"em_3": "em_true"})
         
    old_df = old_df[['em_true','va_usd','employees']]
    old_df = old_df.rename(columns={"em_true": "old_em_true", "va_usd": "old_va_usd", "employees": "old_employees"})
    
    return old_df
    
    
    
    
    
    
    
    
    
    
    