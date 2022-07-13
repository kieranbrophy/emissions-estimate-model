#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:30:56 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import pandas as pd
from sray_db.pk import PrimaryKey

import input_scoring_toolbox.loading_tools as lt
from sray_db.apps import apps

'''
Load in variables
'''
print("Getting asset IDs...")
assetname_df = lt.get_meta_data(apps['assetinfo_name'][(1,0,0,0)])

master_df = pd.read_csv('model_output/Emissions_Estimate_Output.csv')

HSBC_df = pd.read_csv('clients/HSBC/Names for HSBC emissions sample.csv')
JPM_df = pd.read_csv('clients/JPM/Arabesque Carbon Data Review Updated 20220527.csv')

HSBC_df = HSBC_df.merge(master_df, on = 'assetid',how='inner')
JPM_df = JPM_df.merge(master_df, on = 'assetid',how='inner')

result_df = assetname_df.merge(master_df, left_on = assetname_df.index, right_on = 'assetid')

HSBC_df.to_csv('clients/HSBC/HSBC_updated_new.csv')
JPM_df.to_csv('clients/JPM/JPM_emissions_estimates.csv')