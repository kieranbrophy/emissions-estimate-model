#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:36:24 2022

@author: kieran.brophyarabesque.com
"""
from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

import input_scoring_toolbox.loading_tools as lt
import in_out_functions as iof
from sray_db.apps import apps
from sray_db.apps.pk import PrimaryKey

import eem_regression as eem_regs
import eem_cal_functions as eem
import _config_ as config
import merge_tables as merge_tables

start='2020-01-01'
today = date.today()

'''
Load in variables
'''
print("Getting asset IDs...")
assetid_df = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)])
industry_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], columns = ['industry'])
geography_df = lt.get_meta_data(apps['assetinfo_exch_country'][(1,0,0,0)], columns = ['region','iso2'])
fsymid_df = lt.get_meta_data(apps['assetinfo_security_id'][(1,0,0,0)], columns = ['fsymid']).rename(columns={"fsymid": "fsym_id"})

fsymid_df['asset_id'] = fsymid_df.index

print("Getting emissions...")
emissions_df = iof.load_app_data('temperature_emissions', (1, 1, 0, 0), str(start), str(today))  #************************#

'''
Define which scope we wish to estimate
'''
if config.scope == 'one':
    em_df = emissions_df.dropna(subset = ["em_1"]).rename(columns={"em_1": "em_true"})
elif config.scope == 'two':
    em_df = emissions_df.dropna(subset = ["em_2"]).rename(columns={"em_2": "em_true"})
elif config.scope == 'three':
    em_df = emissions_df.dropna(subset = ["em_3"]).rename(columns={"em_3": "em_true"})

print('Getting revenue...')
tf_df = iof.load_app_data('temperature_financials', (1, 1, 0, 0), '2021-10-01', str(today))  #************************#
tf_df = tf_df[[PrimaryKey.assetid, PrimaryKey.date, 'va_usd','revenue']]

print("Getting market capitalisation...")
mktcap_df = iof.load_app_data('sray_mktcap', (2, 6, 1, 0), str(start), str(today))  #************************#
mktcap_df = mktcap_df[[PrimaryKey.assetid, PrimaryKey.date, 'mktcap_avg_12m']]

print("Getting number of employees...")
esg_df = lt.get_raw_data('esg', str(start), str(today))  #************************#
employees_df = esg_df[['asset_id','35800_reporting_period','35800']]
employees_df = employees_df.rename(columns={'asset_id':PrimaryKey.assetid,'35800_reporting_period':PrimaryKey.date,'35800': "employees"})
employees_df[PrimaryKey.date] = pd.to_datetime(employees_df[PrimaryKey.date])
employees_df = employees_df.dropna(subset = [PrimaryKey.date])

print("Getting other financial data...")
oth_fin_df = fsymid_df.merge(iof.get_csv_from_file('other_fin_data.csv'), on='fsym_id', how='outer').rename(columns={"asset_id": PrimaryKey.assetid, 'date': PrimaryKey.date})
oth_fin_df = oth_fin_df.dropna(subset = [PrimaryKey.assetid])
oth_fin_df[PrimaryKey.assetid] = oth_fin_df[PrimaryKey.assetid].astype('int')
oth_fin_df[PrimaryKey.date] = pd.to_datetime(oth_fin_df[PrimaryKey.date])
oth_fin_df = oth_fin_df.dropna(subset = [PrimaryKey.date])

print("Merging tables...")
info_df = assetid_df.merge(fsymid_df, left_index=True, right_index=True).merge(industry_df, left_index=True, right_index=True).merge(geography_df, left_index=True, right_index=True)    
#info_df[PrimaryKey.assetid] = info_df.index

all_df = merge_tables.mergeTables(info_df, em_df, tf_df, mktcap_df, employees_df, oth_fin_df)

all_df_large = all_df[~(all_df['em_true'] <= 1000)]

'''
Introduce the entities we want to estimate (test_df) and the entities that will train our model (train_df)
'''
test_length = int(0.2*len(all_df_large))

test_df = all_df_large.head(test_length)
train_df = all_df_large.iloc[test_length:]

'''
Single linear regression
'''
import eem_regression as eem_regs
import eem_cal_functions as eem

singleReg_df = eem_regs.singleReg(test_df, train_df)

sx = singleReg_df[PrimaryKey.assetid]
sy = 100*(singleReg_df['em_est'] - singleReg_df['em_true'])/singleReg_df['em_true']

print('Single median error:', sy.median())

import matplotlib.pyplot as plt
plt.scatter(sx, sy)
plt.xlabel('Asset ID')
plt.ylabel('% error')
plt.show()

plt.hist(sy[(sy < 1000)], bins=10)
plt.xlim(-1000, 1000)
plt.xlabel('% error')
plt.ylabel('Number of entities')
plt.show() 

100*len(sy[(sy < 100)])/len(sy)

'''
XG boosting method
'''
import eem_regression as eem_regs
import eem_cal_functions as eem
import boostyBoost as boostyBoost

xgBoost_df = eem_regs.xgBoost(test_df, train_df)

xx = xgBoost_df[PrimaryKey.assetid]
xy = 100*(xgBoost_df['em_est'] - xgBoost_df['em_true'])/xgBoost_df['em_true']

print('XG Boost median error:', xy.median())

import matplotlib.pyplot as plt
plt.scatter(xx, xy)
plt.xlabel('Asset ID')
plt.ylabel('% error')
plt.show()

plt.hist(xy[(xy < 1000)], bins=10)
plt.xlim(-1000, 1000)
plt.xlabel('% error')
plt.ylabel('Number of entities')
plt.show() 

100*len(xy.dropna()[(xy.dropna() < 100)])/len(xy.dropna())
