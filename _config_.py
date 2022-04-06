#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:49:34 2022

@author: kieran.brophyarabesque.com
"""
'''
Define scope of emissions

Can be one, two, three
'''
scope = 'one'

'''
Define minimum number of datapoints that is acceptable to the regression
'''
min_datapoints = 20

'''
Define independent variables in the regression
'''
variables = ['va_usd','revenue','employees','mktcap_avg_12m','ff_assets','ff_eq_tot','ff_mkt_val']

#drop_var = []