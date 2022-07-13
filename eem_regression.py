#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:12:51 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_prod")

import pandas as pd

from sklearn.model_selection import train_test_split

import eem_cal_functions as eem
import _config_ as config
from sray_db.pk import PrimaryKey

from joblib import dump
from joblib import load

def industry(all_df, scope, run) -> pd.DataFrame:
    
    '''
    Introduce categorical data
    '''
    all_df[['industry']] = all_df.industry.astype('category')
    all_df[['eco_sector']] = all_df.bespoke_economic_sector.astype('category')
    all_df[['factset_sector']] = all_df.bespoke_industry.astype('category')    
    all_df[['region']] = all_df.region.astype('category')
    all_df[['iso2']] = all_df.iso2.astype('category')
    
    all_df['ind_code'] = all_df.industry.cat.codes
    all_df['eco_code']= all_df.eco_sector.cat.codes
    all_df['fact_code']= all_df.factset_sector.cat.codes
    all_df['reg_code'] = all_df.region.cat.codes
    all_df['iso_code'] = all_df.iso2.cat.codes
    
    all_df = all_df.merge(pd.get_dummies(all_df[['industry']]), on=PrimaryKey.assetid)
    all_df = all_df.merge(pd.get_dummies(all_df[['region']]), on=PrimaryKey.assetid)
    all_df = all_df.merge(pd.get_dummies(all_df[['iso2']]), on=PrimaryKey.assetid)
    
    '''
    Split data in train, test and validation
    '''
    train_df = all_df.dropna(subset = ['em_true'])
    train_df = train_df[(train_df['em_true'] > config.em_thresh)]
        
    X = train_df.drop(columns=['em_true'])
    y = train_df[['fact_code','eco_code','ind_code','reg_code','iso_code','em_true']]
    
    '''
    Use same test and validation set as used before?
    '''
    if config.use_prev_locs == True:
        
        X_val_loc = load('inputs/run_' + str(run) + '/X_train_set_scope_' + str(scope) + '.dat')
        X_val = X[X.index.isin(X_val_loc[PrimaryKey.assetid])]
        y_val = y[y.index.isin(X_val_loc[PrimaryKey.assetid])]
        
        X_test_loc = load('inputs/run_' + str(run) + '/X_test_set_scope_' + str(scope) + '.dat')
        X_test = X[X.index.isin(X_test_loc[PrimaryKey.assetid])]
        y_test = y[y.index.isin(X_test_loc[PrimaryKey.assetid])]
        
        X_train_test = X[~X.index.isin(X_test.index)]
        X_train_test = X_train_test[~X_train_test.index.isin(X_val.index)]
        y_train = y[~y.index.isin(y_test.index)]
        y_train = y_train[~y_train.index.isin(y_val.index)]
        
    else:
        
        X_train, X_test_tot, y_train, y_test_tot = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y['fact_code'], shuffle=True)
        
        if scope == 'three':
            X_val, X_test, y_val, y_test = train_test_split(X_test_tot, y_test_tot, test_size=0.5, random_state=0, stratify=y_test_tot['eco_code'], shuffle=True)
        else:
            X_val, X_test, y_val, y_test = train_test_split(X_test_tot, y_test_tot, test_size=0.5, random_state=0, stratify=y_test_tot['fact_code'], shuffle=True)
        
    X_real = all_df.drop(columns=['em_true'])
    
    '''
    Save input for future reference
    '''
    if config.save_inputs == True:
    
        y_train.to_csv('inputs/run_' + str(run) + '/y_train_set_scope_' + str(scope) + '.csv')
        dump(y_train, 'inputs/run_' + str(run) + '/y_train_set_scope_' + str(scope) + '.dat')
        
        y_test.to_csv('inputs/run_' + str(run) + '/y_test_set_scope_' + str(scope) + '.csv')
        dump(y_test, 'inputs/run_' + str(run) + '/y_test_set_scope_' + str(scope) + '.dat')
        
        y_val.to_csv('inputs/run_' + str(run) + '/y_val_set_scope_' + str(scope) + '.csv')
        dump(y_val, 'inputs/run_' + str(run) + '/y_val_set_scope_' + str(scope) + '.dat')
    
        X_train.to_csv('inputs/run_' + str(run) + '/X_train_set_scope_' + str(scope) + '.csv')
        dump(X_train, 'inputs/run_' + str(run) + '/X_train_set_scope_' + str(scope) + '.dat')
        
        X_test.to_csv('inputs/run_' + str(run) + '/X_test_set_scope_' + str(scope) + '.csv')
        dump(X_test, 'inputs/run_' + str(run) + '/X_test_set_scope_' + str(scope) + '.dat')
        
        X_val.to_csv('inputs/run_' + str(run) + '/X_val_set_scope_' + str(scope) + '.csv')
        dump(X_val, 'inputs/run_' + str(run) + '/X_val_set_scope_' + str(scope) + '.dat')
        
        X_real.to_csv('inputs/run_' + str(run) + '/X_real_scope_' + str(scope) + '.csv')
        dump(X_real, 'inputs/run_' + str(run) + '/X_real_set_scope_' + str(scope) + '.dat')
        
    
    '''
    XG Boost method
    '''
    BigBoosty = X_train.groupby('eco_code').apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['eco_code'] == x['eco_code'].iloc[0]],
                                                                              X_test.loc[X_test['eco_code'] == x['eco_code'].iloc[0]], y_test.loc[y_test['eco_code'] == x['eco_code'].iloc[0]],
                                                                              X_val.loc[X_val['eco_code'] == x['eco_code'].iloc[0]], y_val.loc[y_val['eco_code'] == x['eco_code'].iloc[0]],
                                                                              X_real.loc[X_real['eco_code'] == x['eco_code'].iloc[0]], scope, run, x['eco_code'].iloc[0], 0, 0))
    
    BigBoosty = BigBoosty.reset_index('eco_code', drop=True)
    
    '''
    Optimise for Combined FactSet sector
    '''
    BigBoosty_Fact = X_train.groupby('fact_code').apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['fact_code'] == x['fact_code'].iloc[0]],
                                                                                   X_test.loc[X_test['fact_code'] == x['fact_code'].iloc[0]], y_test.loc[y_test['fact_code'] == x['fact_code'].iloc[0]],
                                                                                   X_val.loc[X_val['fact_code'] == x['fact_code'].iloc[0]], y_val.loc[y_val['fact_code'] == x['fact_code'].iloc[0]],
                                                                                   X_real.loc[X_real['fact_code'] == x['fact_code'].iloc[0]], scope, run, x['fact_code'].iloc[0], 0, 1))
        
            
    BigBoosty_Fact = BigBoosty_Fact.reset_index('fact_code', drop=True)
            
    BigBoosty_FactM = BigBoosty.merge(BigBoosty_Fact, on=PrimaryKey.assetid, how='outer')
        
    for entity in BigBoosty.index:
        if BigBoosty_FactM['XGB_R_squared_y'][entity] > BigBoosty_FactM['XGB_R_squared_x'][entity] and BigBoosty_FactM['XGB_per_error_est_y'][entity] < BigBoosty_FactM['XGB_per_error_est_x'][entity]:
            BigBoosty['XGB_em_est'][entity] = BigBoosty_FactM['XGB_em_est_y'][entity]
            BigBoosty['XGB_error_est'][entity] = BigBoosty_FactM['XGB_error_est_y'][entity]
            BigBoosty['XGB_per_error_est'][entity] = BigBoosty_FactM['XGB_per_error_est_y'][entity]
            BigBoosty['XGB_R_squared'][entity] = BigBoosty_FactM['XGB_R_squared_y'][entity]
            BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_FactM['XGB_datapoints_val_y'][entity]
            BigBoosty['XGB_model_file'][entity] = BigBoosty_FactM['XGB_model_file_y'][entity]
                    
    '''
    Optimise for Industry
    '''        
    BigBoosty_Ind = X_train.groupby('ind_code').apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[y_train['ind_code'] == x['ind_code'].iloc[0]],
                                                                                  X_test.loc[X_test['ind_code'] == x['ind_code'].iloc[0]], y_test.loc[y_test['ind_code'] == x['ind_code'].iloc[0]],
                                                                                  X_val.loc[X_val['ind_code'] == x['ind_code'].iloc[0]], y_val.loc[y_val['ind_code'] == x['ind_code'].iloc[0]],
                                                                                  X_real.loc[X_real['ind_code'] == x['ind_code'].iloc[0]], scope, run, x['ind_code'].iloc[0], 0, 2))
                    
    BigBoosty_Ind = BigBoosty_Ind.reset_index('ind_code', drop=True)
            
    BigBoosty_IndM = BigBoosty.merge(BigBoosty_Ind, on=PrimaryKey.assetid, how='outer')
        
    for entity in BigBoosty.index:
        if BigBoosty_IndM['XGB_R_squared_y'][entity] > BigBoosty_IndM['XGB_R_squared_x'][entity] and BigBoosty_IndM['XGB_per_error_est_y'][entity] < BigBoosty_IndM['XGB_per_error_est_x'][entity]:
            BigBoosty['XGB_em_est'][entity] = BigBoosty_IndM['XGB_em_est_y'][entity]
            BigBoosty['XGB_error_est'][entity] = BigBoosty_IndM['XGB_error_est_y'][entity]
            BigBoosty['XGB_per_error_est'][entity] = BigBoosty_IndM['XGB_per_error_est_y'][entity]
            BigBoosty['XGB_R_squared'][entity] = BigBoosty_IndM['XGB_R_squared_y'][entity]
            BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_IndM['XGB_datapoints_val_y'][entity]
            BigBoosty['XGB_model_file'][entity] = BigBoosty_IndM['XGB_model_file_y'][entity]
            
    '''
    Optimise for geography and sector
    '''
    '''
    Region and economic sector
    '''
    if config.geo_opt == True:
        BigBoosty_EcoReg = X_train.groupby(['eco_code','reg_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['reg_code'] == x['reg_code'].iloc[0]) & (y_train['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['reg_code'] == x['reg_code'].iloc[0]) & (X_test['eco_code'] == x['eco_code'].iloc[0])], y_test.loc[(y_test['reg_code'] == x['reg_code'].iloc[0]) & (y_test['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['reg_code'] == x['reg_code'].iloc[0]) & (X_val['eco_code'] == x['eco_code'].iloc[0])], y_val.loc[(y_val['reg_code'] == x['reg_code'].iloc[0]) & (y_val['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['reg_code'] == x['reg_code'].iloc[0]) & (X_real['eco_code'] == x['eco_code'].iloc[0])], scope, run, x['eco_code'].iloc[0], x['reg_code'].iloc[0], 3))
        if len(BigBoosty_EcoReg) > 0:
            BigBoosty_EcoReg = BigBoosty_EcoReg.reset_index(['eco_code','reg_code'], drop=True)
        
            BigBoosty_EcoRegM = BigBoosty.merge(BigBoosty_EcoReg, on=PrimaryKey.assetid, how='outer')
        
            for entity in BigBoosty.index:
                if BigBoosty_EcoRegM['XGB_R_squared_y'][entity] > BigBoosty_EcoRegM['XGB_R_squared_x'][entity] and BigBoosty_EcoRegM['XGB_per_error_est_y'][entity] < BigBoosty_EcoRegM['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_EcoRegM['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_EcoRegM['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_EcoRegM['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_EcoRegM['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_EcoRegM['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_EcoRegM['XGB_model_file_y'][entity]
         
        '''
        Country and economic sector
        '''
        BigBoosty_EcoIso = X_train.groupby(['eco_code','iso_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['iso_code'] == x['iso_code'].iloc[0]) & (y_train['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['iso_code'] == x['iso_code'].iloc[0]) & (X_test['eco_code'] == x['eco_code'].iloc[0])], y_test.loc[(y_test['iso_code'] == x['iso_code'].iloc[0]) & (y_test['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['iso_code'] == x['iso_code'].iloc[0]) & (X_val['eco_code'] == x['eco_code'].iloc[0])], y_val.loc[(y_val['iso_code'] == x['iso_code'].iloc[0]) & (y_val['eco_code'] == x['eco_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['iso_code'] == x['iso_code'].iloc[0]) & (X_real['eco_code'] == x['eco_code'].iloc[0])], scope, run, x['eco_code'].iloc[0], x['iso_code'].iloc[0], 4))
        
        if len(BigBoosty_EcoIso) > 0:
            
            BigBoosty_EcoIso = BigBoosty_EcoIso.reset_index(['eco_code','iso_code'], drop=True)
        
            BigBoosty_EcoIsoM = BigBoosty.merge(BigBoosty_EcoIso, on=PrimaryKey.assetid, how='outer')
        
            for entity in BigBoosty.index:
                if BigBoosty_EcoIsoM['XGB_R_squared_y'][entity] > BigBoosty_EcoIsoM['XGB_R_squared_x'][entity] and BigBoosty_EcoIsoM['XGB_per_error_est_y'][entity] < BigBoosty_EcoIsoM['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_EcoIsoM['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_EcoIsoM['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_EcoIsoM['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_EcoIsoM['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_EcoIsoM['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_EcoIsoM['XGB_model_file_y'][entity]
                
        '''
        Region and Combined FactSet sector
        '''
        BigBoosty_FactReg = X_train.groupby(['fact_code','reg_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['reg_code'] == x['reg_code'].iloc[0]) & (y_train['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['reg_code'] == x['reg_code'].iloc[0]) & (X_test['fact_code'] == x['fact_code'].iloc[0])], y_test.loc[(y_test['reg_code'] == x['reg_code'].iloc[0]) & (y_test['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['reg_code'] == x['reg_code'].iloc[0]) & (X_val['fact_code'] == x['fact_code'].iloc[0])], y_val.loc[(y_val['reg_code'] == x['reg_code'].iloc[0]) & (y_val['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['reg_code'] == x['reg_code'].iloc[0]) & (X_real['fact_code'] == x['fact_code'].iloc[0])], scope, run, x['fact_code'].iloc[0], x['reg_code'].iloc[0], 5))
        if len(BigBoosty_FactReg) > 0:
            BigBoosty_FactReg = BigBoosty_FactReg.reset_index(['fact_code','reg_code'], drop=True)
        
            BigBoosty_FactRegM = BigBoosty.merge(BigBoosty_FactReg, on=PrimaryKey.assetid, how='outer')
        
            for entity in BigBoosty.index:
                if BigBoosty_FactRegM['XGB_R_squared_y'][entity] > BigBoosty_FactRegM['XGB_R_squared_x'][entity] and BigBoosty_FactRegM['XGB_per_error_est_y'][entity] < BigBoosty_FactRegM['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_FactRegM['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_FactRegM['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_FactRegM['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_FactRegM['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_FactRegM['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_FactRegM['XGB_model_file_y'][entity]
        
        '''
        Country and Combined FactSet sector
        '''
        BigBoosty_FactIso = X_train.groupby(['fact_code','iso_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['iso_code'] == x['iso_code'].iloc[0]) & (y_train['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['iso_code'] == x['iso_code'].iloc[0]) & (X_test['fact_code'] == x['fact_code'].iloc[0])], y_test.loc[(y_test['iso_code'] == x['iso_code'].iloc[0]) & (y_test['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['iso_code'] == x['iso_code'].iloc[0]) & (X_val['fact_code'] == x['fact_code'].iloc[0])], y_val.loc[(y_val['iso_code'] == x['iso_code'].iloc[0]) & (y_val['fact_code'] == x['fact_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['iso_code'] == x['iso_code'].iloc[0]) & (X_real['fact_code'] == x['fact_code'].iloc[0])], scope, run, x['fact_code'].iloc[0], x['iso_code'].iloc[0], 6))
        
        if len(BigBoosty_FactIso) > 0:
            BigBoosty_FactIso = BigBoosty_FactIso.reset_index(['fact_code','iso_code'], drop=True)
            
            BigBoosty_FactIsoM = BigBoosty.merge(BigBoosty_FactIso, on=PrimaryKey.assetid, how='outer')
            
            for entity in BigBoosty.index:
                if BigBoosty_FactIsoM['XGB_R_squared_y'][entity] > BigBoosty_FactIsoM['XGB_R_squared_x'][entity] and BigBoosty_FactIsoM['XGB_per_error_est_y'][entity] < BigBoosty_FactIsoM['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_FactIsoM['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_FactIsoM['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_FactIsoM['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_FactIsoM['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_FactIsoM['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_FactIsoM['XGB_model_file_y'][entity]
                    
        
        '''
        Region and Industry
        '''
        BigBoosty_IndReg = X_train.groupby(['ind_code','reg_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['reg_code'] == x['reg_code'].iloc[0]) & (y_train['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['reg_code'] == x['reg_code'].iloc[0]) & (X_test['ind_code'] == x['ind_code'].iloc[0])], y_test.loc[(y_test['reg_code'] == x['reg_code'].iloc[0]) & (y_test['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['reg_code'] == x['reg_code'].iloc[0]) & (X_val['ind_code'] == x['ind_code'].iloc[0])], y_val.loc[(y_val['reg_code'] == x['reg_code'].iloc[0]) & (y_val['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['reg_code'] == x['reg_code'].iloc[0]) & (X_real['ind_code'] == x['ind_code'].iloc[0])], scope, run, x['ind_code'].iloc[0], x['reg_code'].iloc[0], 7))
        if len(BigBoosty_IndReg) > 0:
            BigBoosty_IndReg = BigBoosty_IndReg.reset_index(['ind_code','reg_code'], drop=True)
        
            BigBoosty_IndRegM = BigBoosty.merge(BigBoosty_IndReg, on=PrimaryKey.assetid, how='outer')
        
            for entity in BigBoosty.index:
                if BigBoosty_IndRegM['XGB_R_squared_y'][entity] > BigBoosty_IndRegM['XGB_R_squared_x'][entity] and BigBoosty_IndRegM['XGB_per_error_est_y'][entity] < BigBoosty_IndRegM['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_IndRegM['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_IndRegM['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_IndRegM['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_IndRegM['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_IndRegM['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_IndRegM['XGB_model_file_y'][entity]
        
        '''
        Country and Industry
        '''
        BigBoosty_IndIso = X_train.groupby(['ind_code','iso_code']).apply(lambda x: eem.xgBoostyBoost(x, y_train.loc[(y_train['iso_code'] == x['iso_code'].iloc[0]) & (y_train['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_test.loc[(X_test['iso_code'] == x['iso_code'].iloc[0]) & (X_test['ind_code'] == x['ind_code'].iloc[0])], y_test.loc[(y_test['iso_code'] == x['iso_code'].iloc[0]) & (y_test['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_val.loc[(X_val['iso_code'] == x['iso_code'].iloc[0]) & (X_val['ind_code'] == x['ind_code'].iloc[0])], y_val.loc[(y_val['iso_code'] == x['iso_code'].iloc[0]) & (y_val['ind_code'] == x['ind_code'].iloc[0])],
                                                                                 X_real.loc[(X_real['iso_code'] == x['iso_code'].iloc[0]) & (X_real['ind_code'] == x['ind_code'].iloc[0])], scope, run, x['ind_code'].iloc[0], x['iso_code'].iloc[0], 8))
        
        if len(BigBoosty_IndIso) > 0:
            BigBoosty_IndIso = BigBoosty_IndIso.reset_index(['ind_code','iso_code'], drop=True)
            
            BigBoosty_BIG = BigBoosty.merge(BigBoosty_IndIso, on=PrimaryKey.assetid, how='outer')
            
            for entity in BigBoosty.index:
                if BigBoosty_BIG['XGB_R_squared_y'][entity] > BigBoosty_BIG['XGB_R_squared_x'][entity] and BigBoosty_BIG['XGB_per_error_est_y'][entity] < BigBoosty_BIG['XGB_per_error_est_x'][entity]:
                    BigBoosty['XGB_em_est'][entity] = BigBoosty_BIG['XGB_em_est_y'][entity]
                    BigBoosty['XGB_error_est'][entity] = BigBoosty_BIG['XGB_error_est_y'][entity]
                    BigBoosty['XGB_per_error_est'][entity] = BigBoosty_BIG['XGB_per_error_est_y'][entity]
                    BigBoosty['XGB_R_squared'][entity] = BigBoosty_BIG['XGB_R_squared_y'][entity]
                    BigBoosty['XGB_datapoints_val'][entity] = BigBoosty_BIG['XGB_datapoints_val_y'][entity]
                    BigBoosty['XGB_model_file'][entity] = BigBoosty_BIG['XGB_model_file_y'][entity]
    
    result = BigBoosty
    
    return result
