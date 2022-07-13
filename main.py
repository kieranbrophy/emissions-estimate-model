#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:36:24 2022

@author: kieran.brophyarabesque.com
"""
from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_prod")

from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd

import input_scoring_toolbox.loading_tools as lt
from sray_db.apps import apps
from sray_db.pk import PrimaryKey

import eem_regression as eem_regs
import _config_ as config
import merge_tables as merge_tables

today = date.today()
start = today - relativedelta(years=2)

'''
Load in variables
'''
    
print("Getting asset IDs...")
assetid_df = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)])
industry_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], columns = ['industry','economic_sector']).reset_index()
ind_df = industry_df.merge(pd.read_csv('economic_to_industry_mapping.csv')[['industry', 'bespoke_economic_sector', 'bespoke_industry']], on='industry', how='inner').set_index([PrimaryKey.assetid])
geography_df = lt.get_meta_data(apps['assetinfo_exch_country'][(1,0,0,0)], columns = ['region','iso2'])
fsymid_df = lt.get_meta_data(apps['assetinfo_security_id'][(1,0,0,0)], columns = ['fsymid']).rename(columns={"fsymid": "fsym_id"})

fsymid_df['asset_id'] = fsymid_df.index

print("Getting emissions...")
emissions_df = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], '2022-07-03', '2022-07-03')

print("Getting old emissions...")
#oldEm_df = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], '2010-01-01', str(today)) # takes long to load
oldEm_df = pd.read_csv('inputs/app_data/old_emissions.csv').rename(columns={"assetid": PrimaryKey.assetid, 'em_date': PrimaryKey.date})
oldEm_df = oldEm_df.dropna(subset = [PrimaryKey.assetid])
oldEm_df[PrimaryKey.assetid] = oldEm_df[PrimaryKey.assetid].astype('int')
oldEm_df[PrimaryKey.date] = pd.to_datetime(oldEm_df[PrimaryKey.date])
oldEm_df = oldEm_df.dropna(subset = [PrimaryKey.date])
    
print('Getting revenue...')
tf_df = lt.get_app_data(apps['temperature_financials'][(1, 1, 0, 0)], '2022-07-03', '2022-07-03')
tf_df = tf_df[['va_usd','revenue']]

print("Getting old revenue...")
#oldtf_df = lt.get_app_data(apps['temperature_financials'][(1, 1, 0, 0)], '2010-01-01', str(today), freq = 'y') # takes long to load
oldtf_df = pd.read_csv('inputs/app_data/old_financials.csv').rename(columns={"assetid": PrimaryKey.assetid, 'date': PrimaryKey.date})
oldtf_df = oldtf_df.dropna(subset = [PrimaryKey.assetid])
oldtf_df[PrimaryKey.assetid] = oldtf_df[PrimaryKey.assetid].astype('int')
oldtf_df[PrimaryKey.date] = pd.to_datetime(oldtf_df[PrimaryKey.date])
oldtf_df = oldtf_df.dropna(subset = [PrimaryKey.date])

print("Getting ESG data...")
esg_df = lt.get_raw_data('esg', str(start), str(today))

employees_df = esg_df[['asset_id','s_35800_reporting_period','s_35800_input']]
employees_df = employees_df.rename(columns={'asset_id':PrimaryKey.assetid,'s_35800_reporting_period':PrimaryKey.date,'s_35800_input': "employees"})
employees_df[PrimaryKey.date] = pd.to_datetime(employees_df[PrimaryKey.date])
employees_df = employees_df.dropna(subset = [PrimaryKey.date])
    
energy_df = esg_df[['asset_id','e_30200_reporting_period','e_30200_input']]
energy_df = energy_df.rename(columns={'asset_id':PrimaryKey.assetid,'e_30200_reporting_period':PrimaryKey.date,'e_30200_input': "energy"})
energy_df[PrimaryKey.date] = pd.to_datetime(energy_df[PrimaryKey.date])
energy_df = energy_df.dropna(subset = [PrimaryKey.date])
    
ghg_df = esg_df[['asset_id','e_26800_reporting_period','e_26800_input']]
ghg_df = ghg_df.rename(columns={'asset_id':PrimaryKey.assetid,'e_26800_reporting_period':PrimaryKey.date,'e_26800_input': "ghg"})
ghg_df[PrimaryKey.date] = pd.to_datetime(ghg_df[PrimaryKey.date])
ghg_df = ghg_df.dropna(subset = [PrimaryKey.date])
    
print("Getting other financial data...")
oth_fin_df = fsymid_df.merge(pd.read_csv('inputs/factset_fun/other_fin_data_3.csv'), on='fsym_id', how='outer').rename(columns={"asset_id": PrimaryKey.assetid, 'date': PrimaryKey.date})
oth_fin_df = oth_fin_df.dropna(subset = [PrimaryKey.assetid])
oth_fin_df[PrimaryKey.assetid] = oth_fin_df[PrimaryKey.assetid].astype('int')
oth_fin_df[PrimaryKey.date] = pd.to_datetime(oth_fin_df[PrimaryKey.date])
oth_fin_df = oth_fin_df.dropna(subset = [PrimaryKey.date])
    
run_label = ['20220703']
scopes_label = ['one','two','three']

for run in run_label:
    for scope in scopes_label:
    
        '''
        Define which scope we wish to estimate
        '''
        if scope == 'one':
            em_df = emissions_df.dropna(subset = ["em_1"]).rename(columns={"em_1": "em_true"})
        elif scope == 'two':
            em_df = emissions_df.dropna(subset = ["em_2"]).rename(columns={"em_2": "em_true"})
        elif scope == 'three':
            em_df = emissions_df.dropna(subset = ["em_3"]).rename(columns={"em_3": "em_true"})
        
        '''
        Include old emissions
        '''
        old_em_df = merge_tables.oldEm(oldEm_df, oldtf_df, employees_df, scope)

        print("Merging tables...")
        info_df = assetid_df.merge(fsymid_df, left_index=True, right_index=True).merge(ind_df, left_index=True, right_index=True).merge(geography_df, left_index=True, right_index=True)

        all_df = merge_tables.mergeTables(info_df, em_df, tf_df, employees_df, energy_df, ghg_df, oth_fin_df)

        '''
        Company specific emissions estimate model
        '''
        scaling_va = all_df['va_usd']/old_em_df['old_va_usd']
        scaling_emp = all_df['employees']/old_em_df['old_employees']

        company_va = old_em_df['old_em_true']*scaling_va.drop_duplicates()
        company_emp = old_em_df['old_em_true']*scaling_emp.drop_duplicates()

        company_s = pd.concat([company_va, company_emp], axis=1).mean(axis=1)
        company_df = pd.DataFrame({'sca_em_est': company_s})

        '''
        If we wish to incorporate results of company emissions model in industry specific model
        '''
        if config.scale_old_em == True:
            all_df['em_true'] = all_df['em_true'].fillna(company_df['sca_em_est'])
    
        '''
        Industry specific emissions estimate model - using XG boosting method
        ''' 
        industry_df = eem_regs.industry(all_df, scope, run)

        '''
        Combine Company and industry models
        '''
        result_df = all_df.merge(industry_df[['XGB_em_est', 'XGB_per_error_est', 'XGB_R_squared','XGB_datapoints_val','XGB_model_file',
                                      'em_val']],
                                      on=PrimaryKey.assetid, how='outer').merge(company_df,on=PrimaryKey.assetid,how='outer').dropna(subset=['industry'])

        '''
        Save results
        '''
        result_df.to_csv('model_results/run_' + str(run) + '/scope_' + str(scope) + '.csv')
