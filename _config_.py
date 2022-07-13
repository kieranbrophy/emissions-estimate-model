#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:49:34 2022

@author: kieran.brophyarabesque.com
"""
'''
Sector and geographical optimisation?
'''
geo_opt = True

'''
Fill gaps in emissions data with scaling previous emissions data?
'''
scale_old_em = True

'''
Threshold emissions for training data
'''
em_thresh = 100

'''
Threshold validation datapoints
'''
data_thresh = 3

'''
Threshold amount of non-nan's in training data (out of 10)
'''
nan_thresh = 6

'''
Run hyperparamter tuning on XG Boost?
'''
hyper_tune = True

'''
Define number of trials to tune XG boost hyperparameters
'''
n_trials = 30

'''
Define independent variables in the regression
'''
variables_long = ['industry','economic_sector','bespoke_economic_sector','bespoke_industry','region','iso2',
                     'em_true', 'va_usd', 'employees', 'energy', 'ghg', 'ff_assets', 'ff_eq_tot', 'ff_mkt_val',
                     'ff_sales','ff_gross_inc','ff_oper_exp_tot']

variables_short = ['va_usd', 'employees', 'energy', 'ghg', 'ff_assets', 'ff_eq_tot', 'ff_mkt_val',
                     'ff_sales','ff_gross_inc','ff_oper_exp_tot']
'''
Shap? - XG Booster needs to be set to 'GBtree' is True
'''
show_shap = True

'''
Show optimsied trial plots?
'''
show_opt_trial = True

'''
Use previously used validation and test set assetids?
'''
use_prev_locs = False

'''
Save inputs?
'''
save_inputs = True