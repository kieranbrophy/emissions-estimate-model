#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:35:53 2022

@author: kieran.brophyarabesque.com
"""

import pandas as pd

scope1_OG = pd.read_csv('model_results/run_20220701/scope_three.csv')

scope1_2 = pd.read_csv('model_results/run_202207012/scope_three.csv')

scope1_OG = scope1_OG.set_index(['PrimaryKey.assetid'])
scope1_2 = scope1_2.set_index(['PrimaryKey.assetid'])

compare = scope1_OG.compare(scope1_2)

scope1_OG['difference'] = 100*abs(scope1_OG.XGB_em_est - scope1_2.XGB_em_est)/scope1_OG.XGB_em_est

total_diff = scope1_OG['difference'].median()




'''
'''

test = pd.read_csv('test.csv')
test = test.dropna(subset = ['em_3'])