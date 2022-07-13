#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:16:42 2022

@author: kieran.brophyarabesque.com
"""

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

scope1_df = pd.read_csv('model_results/run_20220703/scope_one.csv')
scope2_df = pd.read_csv('model_results/run_20220703/scope_two.csv')
scope3_df = pd.read_csv('model_results/run_20220703/scope_three.csv')

scope1_df['diff'] = [0] * len(scope1_df)
scope2_df['diff'] = [0] * len(scope2_df)
scope3_df['diff'] = [0] * len(scope3_df)

for num in scope1_df.index:
    
    if scope1_df['em_true'][num] != scope1_df['sca_em_est'][num]:
        
        scope1_df['diff'][num] = 100*abs(scope1_df['em_true'][num] - scope1_df['sca_em_est'][num])/scope1_df['em_true'][num]
        
for num in scope2_df.index:
    
    if scope2_df['em_true'][num] != scope2_df['sca_em_est'][num]:
        
        scope2_df['diff'][num] = 100*abs(scope2_df['em_true'][num] - scope2_df['sca_em_est'][num])/scope2_df['em_true'][num]
  
        
for num in scope3_df.index:
    
    if scope3_df['em_true'][num] != scope3_df['sca_em_est'][num]:
        
        scope3_df['diff'][num] = 100*abs(scope3_df['em_true'][num] - scope3_df['sca_em_est'][num])/scope3_df['em_true'][num]
  

med_err1 = np.median(scope1_df.dropna(subset = ['em_val'])['diff'].dropna())
med_err2 = np.median(scope2_df.dropna(subset = ['em_val'])['diff'].dropna())
med_err3 = np.median(scope3_df.dropna(subset = ['em_val'])['diff'].dropna())

scope1_sca = scope1_df.dropna(subset = ['em_val']).dropna()
scope2_sca = scope2_df.dropna(subset = ['em_val']).dropna()
scope3_sca = scope3_df.dropna(subset = ['em_val']).dropna()

r_sq1 = r2_score(scope1_sca['em_true'], scope1_sca['sca_em_est'])
r_sq2 = r2_score(scope2_sca['em_true'], scope2_sca['sca_em_est'])
r_sq3 = r2_score(scope3_sca['em_true'], scope3_sca['sca_em_est'])


