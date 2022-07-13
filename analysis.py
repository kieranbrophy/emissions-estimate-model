#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:37:27 2022

@author: kieran.brophyarabesque.com
"""

import pandas as pd

result_df = pd.read_csv('model_results/run_20220703/scope_three.csv')
#result_df = result_df.loc[result_df.XGB_em_est > 10000]

'''
Analysis
'''
analysis_df = result_df.dropna(subset=['em_true','em_val'])

XGB_per_error = (100*(analysis_df['XGB_em_est'] - analysis_df['em_val'])/analysis_df['em_val'])
XGB_xx = XGB_per_error.index

'''
Print results
'''
XGB1 = XGB_per_error[(XGB_per_error < 100)]
XGB_CORRECT = XGB1[(XGB1 > -50)]
XGB_CORRECT_PER = 100*len(XGB_CORRECT)/len(XGB_per_error)
print('% of XGB estimates within 100% of truth:', XGB_CORRECT_PER)

print('Absolute XGB emissions as % to truth:', 100*analysis_df['XGB_em_est'].sum()/analysis_df['em_val'].sum())

XGB1 = XGB_per_error[(XGB_per_error < 1000)]
XGB_CORRECT = XGB1[(XGB1 > -90)]
XGB_CORRECT_PER = 100*len(XGB_CORRECT)/len(XGB_per_error)
print('% of XGB estimates within 1000% of truth:', XGB_CORRECT_PER)

print('XGB median median error', result_df.XGB_per_error_est.median())

print('XGB max median error', result_df.XGB_per_error_est.max())

print('XGB min median error', result_df.XGB_per_error_est.min())

print('XGB R2 above 0.5:',100*len(analysis_df['XGB_R_squared'].loc[analysis_df['XGB_R_squared'] > 0.5])/len(analysis_df))

print('Global XGB R2 above 0.5:',100*len(result_df['XGB_R_squared'].loc[result_df['XGB_R_squared'] > 0.5])/len(result_df))

'''
Plot results
'''
import matplotlib.pyplot as plt
plt.scatter(XGB_xx, XGB_per_error)
plt.xlabel('Asset ID')
plt.ylabel('% error')
plt.show()

plt.hist(XGB_CORRECT, bins=22)
plt.xlim(-100, 1000)
plt.xlabel('% error')
plt.ylabel('Number of entities')
plt.show() 

'''
Human friendly
'''
human_friendly_result_df = result_df[['sca_em_est', 'XGB_em_est', 'XGB_per_error_est', 'XGB_R_squared']]
human_friendly_result_df['em_best'] = human_friendly_result_df['sca_em_est']
human_friendly_result_df['em_best'] = human_friendly_result_df['em_best'].fillna(human_friendly_result_df['XGB_em_est'])
human_friendly_result_df['em_true'] = result_df['em_true']