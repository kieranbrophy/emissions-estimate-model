#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:40:07 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import pandas as pd
from sray_db.apps.pk import PrimaryKey

import input_scoring_toolbox.loading_tools as lt
from sray_db.apps import apps

import math

'''
Load in variables
'''
print("Getting asset IDs...")

assetname_df = lt.get_meta_data(apps['assetinfo_name'][(1,0,0,0)])


scope1_df = pd.read_csv('model_results/run_20220703/scope_one.csv')
scope2_df = pd.read_csv('model_results/run_20220703/scope_two.csv')
scope3_df = pd.read_csv('model_results/run_20220703/scope_three.csv')

merge_df = scope1_df.merge(scope2_df, on=['PrimaryKey.assetid'], how='outer').merge(scope3_df, on=['PrimaryKey.assetid'], how='outer')

bruce_df = merge_df[['PrimaryKey.assetid','XGB_model_file_x','XGB_model_file_y','XGB_model_file']]
bruce_df.to_csv('model_output/XGB_model_mapping.csv')

master_df = merge_df[['PrimaryKey.assetid'
                      ,'em_true_x','sca_em_est_x','XGB_em_est_x','XGB_per_error_est_x','XGB_R_squared_x'
                      ,'em_true_y','sca_em_est_y','XGB_em_est_y','XGB_per_error_est_y','XGB_R_squared_y'
                      ,'em_true','sca_em_est','XGB_em_est','XGB_per_error_est','XGB_R_squared']].rename(columns={'PrimaryKey.assetid':'assetid'})


result_df = assetname_df.merge(master_df, left_on = assetname_df.index, right_on = 'assetid')

'''
Scope 1
'''
result_df['Scope 1 emissions (tCO2e)'] = result_df['em_true_x']
result_df['Scope 1 emissions type'] = 'Disclosed'
result_df['Scope 1 estimate confidence'] = '-'

for num in result_df.index:
    
    if math.isnan(result_df['em_true_x'][num]):  
        
        result_df['Scope 1 emissions type'][num] = 'Estimated'
        
        if math.isnan(result_df['sca_em_est_x'][num]):
            
            result_df['Scope 1 emissions (tCO2e)'][num] = round(result_df['Scope 1 emissions (tCO2e)'][num].fillna(result_df['XGB_em_est_x'][num]))
        
            if result_df.XGB_R_squared_x[num] >= 0.5 and result_df.XGB_per_error_est_x[num] <= 50:
                result_df['Scope 1 estimate confidence'][num] = 'Very High'
            elif result_df.XGB_R_squared_x[num] >= 0.5 and result_df.XGB_per_error_est_x[num] > 50 and result_df.XGB_per_error_est_x[num] <= 100:
                result_df['Scope 1 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared_x[num] >= 0.5 and result_df.XGB_per_error_est_x[num] > 100:
                result_df['Scope 1 estimate confidence'][num] = 'Medium'    
            elif result_df.XGB_R_squared_x[num] < 0.5 and result_df.XGB_R_squared_x[num] >= 0 and result_df.XGB_per_error_est_x[num] <= 50:
                result_df['Scope 1 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared_x[num] < 0.5 and result_df.XGB_R_squared_x[num] >= 0 and result_df.XGB_per_error_est_x[num] > 50 and result_df.XGB_per_error_est_x[num] <= 100:
                result_df['Scope 1 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared_x[num] < 0.5 and result_df.XGB_R_squared_x[num] >= 0 and result_df.XGB_per_error_est_x[num] > 100:
                result_df['Scope 1 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared_x[num] < 0 and result_df.XGB_per_error_est_x[num] <= 50:
                result_df['Scope 1 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared_x[num] < 0 and result_df.XGB_per_error_est_x[num] > 50 and result_df.XGB_per_error_est_x[num] <= 100:
                result_df['Scope 1 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared_x[num] < 0 and result_df.XGB_per_error_est_x[num] > 100:
                result_df['Scope 1 estimate confidence'][num] = 'Very Low'

        else:
        
            result_df['Scope 1 emissions (tCO2e)'][num] = round(result_df['Scope 1 emissions (tCO2e)'][num].fillna(result_df['sca_em_est_x'][num]))
            result_df['Scope 1 estimate confidence'][num] = 'Very High'
        
'''
Scope 2
'''
result_df['Scope 2 emissions (tCO2e)'] = result_df['em_true_y']
result_df['Scope 2 emissions (tCO2e)'] = round(result_df['Scope 2 emissions (tCO2e)'].fillna(result_df['sca_em_est_y']))
result_df['Scope 2 emissions (tCO2e)'] = round(result_df['Scope 2 emissions (tCO2e)'].fillna(result_df['XGB_em_est_y']))

result_df['Scope 2 emissions type'] = 'Disclosed'
result_df['Scope 2 estimate confidence'] = '-'

for num in result_df.index:
    
    if math.isnan(result_df['em_true_y'][num]):    
        
        result_df['Scope 2 emissions type'][num] = 'Estimated'
        
        if math.isnan(result_df['sca_em_est_y'][num]):
        
            if result_df.XGB_R_squared_y[num] >= 0.5 and result_df.XGB_per_error_est_y[num] <= 50:
                result_df['Scope 2 estimate confidence'][num] = 'Very High'
            elif result_df.XGB_R_squared_y[num] >= 0.5 and result_df.XGB_per_error_est_y[num] > 50 and result_df.XGB_per_error_est_y[num] <= 100:
                result_df['Scope 2 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared_y[num] >= 0.5 and result_df.XGB_per_error_est_y[num] > 100:
                result_df['Scope 2 estimate confidence'][num] = 'Medium'    
            elif result_df.XGB_R_squared_y[num] < 0.5 and result_df.XGB_R_squared_y[num] >= 0 and result_df.XGB_per_error_est_y[num] <= 50:
                result_df['Scope 2 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared_y[num] < 0.5 and result_df.XGB_R_squared_y[num] >= 0 and result_df.XGB_per_error_est_y[num] > 50 and result_df.XGB_per_error_est_y[num] <= 100:
                result_df['Scope 2 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared_y[num] < 0.5 and result_df.XGB_R_squared_y[num] >= 0 and result_df.XGB_per_error_est_y[num] > 100:
                result_df['Scope 2 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared_y[num] < 0 and result_df.XGB_per_error_est_y[num] <= 50:
                result_df['Scope 2 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared_y[num] < 0 and result_df.XGB_per_error_est_y[num] > 50 and result_df.XGB_per_error_est_y[num] <= 100:
                result_df['Scope 2 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared_y[num] < 0 and result_df.XGB_per_error_est_y[num] > 100:
                result_df['Scope 2 estimate confidence'][num] = 'Very Low'

        else:
        
            result_df['Scope 2 estimate confidence'][num] = 'Very High'
'''
Scope 3
'''
result_df['Scope 3 emissions (tCO2e)'] = result_df['em_true']
result_df['Scope 3 emissions (tCO2e)'] = round(result_df['Scope 3 emissions (tCO2e)'].fillna(result_df['sca_em_est']))
result_df['Scope 3 emissions (tCO2e)'] = round(result_df['Scope 3 emissions (tCO2e)'].fillna(result_df['XGB_em_est']))
        
result_df['Scope 3 emissions type'] = 'Disclosed'
result_df['Scope 3 estimate confidence'] = '-'

for num in result_df.index:
    
    if math.isnan(result_df['em_true'][num]):  
        
        result_df['Scope 3 emissions type'][num] = 'Estimated'
        
        if math.isnan(result_df['sca_em_est'][num]):
        
            if result_df.XGB_R_squared[num] >= 0.5 and result_df.XGB_per_error_est[num] <= 50:
                result_df['Scope 3 estimate confidence'][num] = 'Very High'
            elif result_df.XGB_R_squared[num] >= 0.5 and result_df.XGB_per_error_est[num] > 50 and result_df.XGB_per_error_est[num] <= 100:
                result_df['Scope 3 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared[num] >= 0.5 and result_df.XGB_per_error_est[num] > 100:
                result_df['Scope 3 estimate confidence'][num] = 'Medium'    
            elif result_df.XGB_R_squared[num] < 0.5 and result_df.XGB_R_squared[num] >= 0 and result_df.XGB_per_error_est[num] <= 50:
                result_df['Scope 3 estimate confidence'][num] = 'High'
            elif result_df.XGB_R_squared[num] < 0.5 and result_df.XGB_R_squared[num] >= 0 and result_df.XGB_per_error_est[num] > 50 and result_df.XGB_per_error_est[num] <= 100:
                result_df['Scope 3 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared[num] < 0.5 and result_df.XGB_R_squared[num] >= 0 and result_df.XGB_per_error_est[num] > 100:
                result_df['Scope 3 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared[num] < 0 and result_df.XGB_per_error_est[num] <= 50:
                result_df['Scope 3 estimate confidence'][num] = 'Medium'
            elif result_df.XGB_R_squared[num] < 0 and result_df.XGB_per_error_est[num] > 50 and result_df.XGB_per_error_est[num] <= 100:
                result_df['Scope 3 estimate confidence'][num] = 'Low'
            elif result_df.XGB_R_squared[num] < 0 and result_df.XGB_per_error_est[num] > 100:
                result_df['Scope 3 estimate confidence'][num] = 'Very Low'

        else:
        
            result_df['Scope 3 estimate confidence'][num] = 'Very High'
        

output_df = result_df[['name','assetid',
                      'Scope 1 emissions type','Scope 1 emissions (tCO2e)','Scope 1 estimate confidence',
                      'Scope 2 emissions type','Scope 2 emissions (tCO2e)','Scope 2 estimate confidence',
                      'Scope 3 emissions type','Scope 3 emissions (tCO2e)','Scope 3 estimate confidence']].dropna().reset_index(drop=True)

output_df.to_csv('model_output/Emissions_Estimate_Output.csv')