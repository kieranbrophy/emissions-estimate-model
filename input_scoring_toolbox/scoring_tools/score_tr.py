#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:27:04 2021

@author: roandufeu
"""
# Loading in libraries
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import gc as clean
import numpy as np
from datetime import datetime as dt
import psycopg2
import scipy.stats as stats

from sray_db.apps import apps

from input_scoring_toolbox.loading_tools import loading_tools as lt
from input_scoring_toolbox.loading_tools.db_client.client import DBClient
from input_scoring_toolbox.config import db_config


def score_tr(data_df, start, end):
    """
    Loads appropriate dictionaries and scores TR data according to TR scoring methodology

    Parameters
    ----------
    data_df (pd.DataFrame): DF of acleaned & resampled TR data
    start (str): calculation start date
    end (str): calculation end date

    Returns
    -------
    tr_ranks_df (pd.DataFrame): DF of scored TR metrics

    """
    
    load_cutoff = dt.now()
    
    #load datapoints dict
    dp_dict = lt.load_dict('input_scoring_toolbox/scoring_tools/tr_inputs/dict.txt', 'tr_score', 'tr_datapoints')
    
    #load formula dict
    form_dict = lt.load_dict('input_scoring_toolbox/scoring_tools/tr_inputs/dict.txt', 'tr_score', 'formula')
    
    # load polarity dictionary
    polarity_dict = lt.load_dict('Data/input_scoring_toolbox_metadata.xlsx', 'tr_score', 'polarity', sharepoint_sheetname = 'tr_dict')
    
    # load currency conversion dictionary
    curr_con_dict = lt.load_dict('Data/input_scoring_toolbox_metadata.xlsx', 'tr_score', 'convert', sharepoint_sheetname = 'tr_dict')
    
    # load denominator dictionary
    denom_dict = lt.load_dict('Data/input_scoring_toolbox_metadata.xlsx', 'tr_score', 'denominator', sharepoint_sheetname = 'tr_dict')
    
    # load sector dictionary
    sector_dict = lt.load_dict('Data/input_scoring_toolbox_metadata.xlsx', 'datapoint', 'industries', sharepoint_sheetname = 'tr_sector_dict')
    
    # Load cleaned TR datapoints
    tr_clean_datapoints_df_stored = data_df.copy()
    
    # load revenue data
    fs_rev = load_revenue(start, end, load_cutoff)  
    
    # load industry data
    meta = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], as_of = load_cutoff)
    meta.index.names = ['assetid']
   
    # convert TR datapoints/values to scores
    tr_ranks_df = scoring_function(tr_clean_datapoints_df_stored, 
                fs_rev, meta, dp_dict, form_dict, polarity_dict, denom_dict, 
                curr_con_dict, sector_dict, start, end)                         
    
    del fs_rev, tr_clean_datapoints_df_stored
    clean.collect()
    
    return tr_ranks_df


def scoring_function(tr_clean_datapoints_df_stored, fs_rev_df, meta, dp_dict, form_dict, polarity_dict, denom_dict,
                        curr_con_dict, sector_dict, start, end):
    """
    Creates normalised scores from Thomson Reuters datapoints
    Method:
        * Datapoints that need to be converted to intensity ratios are divided
          by the appropriate denominator
        * Tet answers (eg. Y/Ns are converted to numeric values)
        * The values of negative datapoints are reversed
        * Data is ranked to produce a score between 0 and 100
    Args:
        tr_clean_datapoints_df_stored (Pandas Dataframe): Dataframe of daily, cleaned TR datapoints
        fs_rev (Pandas Dataframe): Dataframe of factset revenue data (TS_F_SALES_A)
        dp_dict (dictionary): Dictionary of datapoints mapped to each score
        form_dict (dictionary): Dictionary of formula type to calculate each score
        polarity_dict (dictionary): Dictionary of polarity of datapoints
        denom_dict (dictionary): Dictionary of denominator type of datapoint to be converted to ratios
        curr_con_dict (dictionary): Dictionary of scores that need converted to USD
        start (string): start date of data
        end (string): end date of data
        
    Returns:
        Pandas Dataframe: Dataframe of normalised Thomson Reuters scores
        :param normalize:
    """
    fs_rev = fs_rev_df.copy()
    
    del fs_rev_df
    clean.collect()

    # create df of denominators
    denom_cols = list(set([col[0] for col in denom_dict.values() if col[0] in tr_clean_datapoints_df_stored.columns]))
    denom_df = tr_clean_datapoints_df_stored[denom_cols]
    denom_df = denom_df.join(fs_rev.drop(columns='report_date'))
    denom_df = denom_df.replace(['nan', 'None', 0], np.nan)
    denom_df = denom_df.apply(pd.to_numeric)

    # add total rev to datapoints
    tr_clean_datapoints_df_stored = tr_clean_datapoints_df_stored.join(fs_rev.drop(columns='report_date'))

    del fs_rev
    clean.collect()

    # only keep datapoints that will be used in final score calculation
    tr_esg_df_norm = pd.DataFrame(index=tr_clean_datapoints_df_stored.index, columns=dp_dict.keys())

    for tr_score in form_dict.keys():
        
        if tr_clean_datapoints_df_stored[dp_dict[tr_score]].isnull().all(axis = None):
            tr_esg_df_norm[tr_score] = np.nan
        else:
            
            # A: if yes then yes else no 
            if form_dict[tr_score][0] == 'A':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].\
                    replace({'Y': 1, 'N': 0, 'NO': 0, 'ISO': 1, 'ISO14000': 1, 'EMS': 1, 'BOTH': 1, 'Both': 1, 'QUANTITATIVE': 1, 'No': 0, 'nan': 0, 'None': 0, None: 0})
                tr_esg_df_norm[tr_score] = tr_esg_df_norm[tr_score].fillna(0)
    
            # B: AND (if yes then yes else no)
            elif form_dict[tr_score][0] == 'B':
                # Get value for each half
                dp_1 = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score][0]] == 'Y')
                dp_2 = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score][1]] == 'Y')
                # combine so YY = 1 , YN = 0.5, NY = 0.5, NN = 0
                tr_esg_df_norm[tr_score] = (dp_1 + dp_2) / 2
    
            # C: No Change - formerly "if yes then yes else NA"
            elif form_dict[tr_score][0] == 'C':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).\
                    replace({'Y': 1, 'N': 0, 'nan': np.nan, 'None': np.nan}).astype(float)
    
            # D: No Change
            elif form_dict[tr_score][0] == 'D':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).\
                    replace({'Y': 1, 'N': 0, 'nan': np.nan, 'None': np.nan}).astype(float)
    
            # E: Ratio
            elif form_dict[tr_score][0] == 'E':
                tr_clean_datapoints_df_stored[dp_dict[tr_score]] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].\
                    astype(str).replace({'nan': np.nan, 'None': np.nan})
                tr_clean_datapoints_df_stored[dp_dict[tr_score]] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].\
                    apply(pd.to_numeric)
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score][0]] / denom_df[denom_dict[tr_score][0]]
                tr_esg_df_norm[tr_score].loc[np.isinf(tr_esg_df_norm[tr_score])] = np.nan
    
            # F: OR (if yes then yes else no)
            elif form_dict[tr_score][0] == 'F':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].\
                    replace({'Y': 1, 'N': 0, 'NO': 0, 'ISO': 1, 'ISO14000': 1, 'EMS': 1, 'BOTH': 1, 'Both': 1, 'QUANTITATIVE': 1, 'No': 0, 'nan': 0, 'None': 0, None: 0}).\
                    max(axis=1)
    
            # G: If numeric then yes else no
            elif form_dict[tr_score][0] == 'G':
                tr_clean_datapoints_df_stored[dp_dict[tr_score]] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].\
                    astype(str).replace({'None': np.nan, 'nan': np.nan, 'No Limit': np.nan})
                tr_esg_df_norm[tr_score] = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score]].apply(pd.to_numeric) >= 0)
    
            # H: If > 0 then yes else no
            elif form_dict[tr_score][0] == 'H':
                tr_esg_df_norm[tr_score] = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score]].apply(pd.to_numeric) > 0)
    
            # I: AND (no change)
            elif form_dict[tr_score][0] == 'I':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].replace(
                    {'Y': 1, 'N': 0, 'nan': np.nan, 'None': np.nan}).mean(axis=1)
    
            # J: SUM
            elif form_dict[tr_score][0] == 'J':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].replace(
                    {'None': np.nan, 'nan': np.nan}).apply(pd.to_numeric).sum(axis=1, skipna=True, min_count=1)
    
            # Others
            elif tr_score == 'tr_16':
                tr_esg_df_norm[tr_score] = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).replace({'None': 0, 'nan': 0, None: 0}).astype(float).sum(axis=1) > 0)
                
            # Others
            elif tr_score == 'tr_70':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored['trd_89'].\
                    replace({'Y': 1, 'N': 0, 'NO': 0, 'ISO': 1, 'ISO14000': 1, 'EMS': 1, 'BOTH': 1, 'Both': 1, 'QUANTITATIVE': 1, 'No': 0, 'nan': 0, 'None': 0, None: 0})
    
            # tr_192
            elif tr_score == 'tr_192':
                tr_clean_datapoints_df_stored['trd_164'] = tr_clean_datapoints_df_stored['trd_164'].astype(str).replace({'None': np.nan, 'nan': np.nan})
                tr_clean_datapoints_df_stored['trd_165'] = tr_clean_datapoints_df_stored['trd_165'].astype(str).replace({'None': np.nan, 'nan': np.nan})
                tr_clean_datapoints_df_stored['trd_166'] = tr_clean_datapoints_df_stored['trd_166'].astype(str).replace({'None': np.nan, 'nan': np.nan})
    
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_164'].astype(float) > 0] = tr_clean_datapoints_df_stored['trd_166'].astype(float)/tr_clean_datapoints_df_stored['trd_164'].astype(float)
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_165'].astype(float) + tr_clean_datapoints_df_stored['trd_164'].astype(float) > 0] = tr_clean_datapoints_df_stored['trd_166'].astype(float)/(tr_clean_datapoints_df_stored['trd_165'].astype(float) + tr_clean_datapoints_df_stored['trd_164'].astype(float))
    
            elif tr_score == 'tr_166':
                dp_1 = tr_clean_datapoints_df_stored['trd_196'].\
                    replace({'Y': 1, 'N': 0, 'NO': 0, 'ISO': 1, 'ISO14000': 1, 'EMS': 1, 'BOTH': 1, 'Both': 1, 'QUANTITATIVE': 1, 'No': 0, 'nan': 0, 'None': 0, None: 0})
                dp_2 = 1 * (tr_clean_datapoints_df_stored['trd_197'].astype(float) > 0)
                tr_esg_df_norm[tr_score] = pd.concat([dp_1, dp_2], axis=1).max(axis=1)
    
            elif tr_score == 'tr_138':
                dp_1 = tr_clean_datapoints_df_stored['trd_204'].replace({'Y': 1, 'N': 0, 'None': 0, 'nan': 0, None: 0})
                dp_2 = 1 * (tr_clean_datapoints_df_stored['trd_202'].astype(str).replace({'None': np.nan, 'nan': np.nan}).notna())
                tr_esg_df_norm[tr_score] = pd.concat([dp_1, dp_2], axis=1).max(axis=1)
    
            elif tr_score == 'tr_165':
                dp_1 = tr_clean_datapoints_df_stored['trd_239'].astype(str).replace({'Y': 1, 'N': 0, 'None': 0, 'nan': 0, None: 0}).astype(float)
                dps = dp_dict[tr_score].copy()
                dps.remove('trd_239')
                dp_2 = tr_clean_datapoints_df_stored[dps].astype(str).replace({'Y': 1, 'N': 0, 'None': 0, 'nan': 0, None:0}).astype(float).max(axis = 1)
                tr_esg_df_norm[tr_score] = (dp_1 + dp_2) / 2
    
            elif tr_score == 'tr_17':
                tr_clean_datapoints_df_stored[dp_dict[tr_score]] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).replace({'None':np.nan, 'nan':np.nan})
                dp1 = 1 * (tr_clean_datapoints_df_stored['trd_333'].astype(float) > 0)
                dp2 = 1 * (tr_clean_datapoints_df_stored['trd_339'].astype(float) > 0)
                tr_esg_df_norm[tr_score] = 1 * (dp1 + dp2 > 0)
    
            elif tr_score == 'tr_249':
                tr_clean_datapoints_df_stored['trd_387'] = tr_clean_datapoints_df_stored['trd_387'].astype(str).replace({'None':np.nan, 'nan':np.nan})
                tr_clean_datapoints_df_stored['trd_389'] = tr_clean_datapoints_df_stored['trd_389'].astype(str).replace({'None':np.nan, 'nan':np.nan})
    
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored['trd_387']
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_387'].isnull()] = tr_clean_datapoints_df_stored['trd_389']
                tr_esg_df_norm[tr_score].loc[tr_esg_df_norm[tr_score].astype(float) <= 0] = np.nan
    
            elif tr_score == 'tr_113':
                tr_esg_df_norm[tr_score] = 1 * (tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).replace({'None': 0, 'nan': 0, None: 0}).astype(float) > 50)
    
            elif tr_score == 'tr_118':
                tr_clean_datapoints_df_stored[dp_dict[tr_score]] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(str).replace({'None': np.nan, 'nan': np.nan})
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].astype(float) * 1000
                
            elif tr_score == 'tr_137':
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored[dp_dict[tr_score]].fillna(0)
    
            elif tr_score == 'tr_173':
                tr_clean_datapoints_df_stored['trd_256'] = tr_clean_datapoints_df_stored['trd_256'].astype(str).replace({'None': np.nan, 'nan': np.nan})
                tr_clean_datapoints_df_stored['trd_249'] = tr_clean_datapoints_df_stored['trd_249'].astype(str).replace({'None': np.nan, 'nan': np.nan})
    
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored['trd_256'].astype(float) / tr_clean_datapoints_df_stored['trd_249'].astype(float)
    
                tr_esg_df_norm[tr_score].loc[tr_esg_df_norm[tr_score].astype(float) < 0] = np.nan
    
                tr_esg_df_norm[tr_score].loc[tr_esg_df_norm[tr_score].astype(float) > 1] = np.nan
    
            elif tr_score == 'tr_200':
                tr_clean_datapoints_df_stored['trd_322'] = tr_clean_datapoints_df_stored['trd_322']. \
                    astype(str).replace({'None': np.nan, 'nan': np.nan})
                dp_1 = 1 * (tr_clean_datapoints_df_stored['trd_322'].apply(pd.to_numeric) >= 0)
                dp_2 = 1 * (tr_clean_datapoints_df_stored['trd_325'] == 'Y')
    
                tr_esg_df_norm[tr_score] = (dp_1 + dp_2) / 2
    
            elif tr_score == 'tr_233':
                tr_clean_datapoints_df_stored['trd_373'] = tr_clean_datapoints_df_stored['trd_373'].astype(str).replace({'None': np.nan, 'nan': np.nan})
                tr_clean_datapoints_df_stored['trd_369'] = tr_clean_datapoints_df_stored['trd_369'].astype(str).replace({'None': np.nan, 'nan': np.nan})
    
                tr_esg_df_norm[tr_score] = tr_clean_datapoints_df_stored['trd_373'].astype(float) / tr_clean_datapoints_df_stored['trd_369'].astype(float)
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_373'].astype(float) == 0] = 0
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_369'].astype(float) == 0] = np.nan
                tr_esg_df_norm[tr_score][tr_clean_datapoints_df_stored['trd_373'].astype(float) + tr_clean_datapoints_df_stored['trd_369'].astype(float) == 0] = 0

    # set start date for loading currency
    start_currency = tr_clean_datapoints_df_stored['report_date'].min()

    # load currency data
    exch_df, curr_df = get_exchange_rates(start_currency, end)
    
    # resample & forward fill by seven days (up to today)
    exch_df = exch_df.set_index(['isocode', 'date'])
    exch_df_resampled = resample(exch_df, freq = 'D', method = 'ffill', limit = 7, id_idx='isocode',
             date_idx='date')
    del exch_df
    clean.collect()

    exch_df_resampled = exch_df_resampled.reset_index()
    
    curr_df = curr_df.set_index('assetid')

    # Match assets with reporting currency code
    exc_rates_df = tr_clean_datapoints_df_stored['report_date'].to_frame().join(curr_df).drop(
        columns=['perusdquotation', 'currencyassetid'])
    exch_df_resampled = exch_df_resampled.rename(columns={'date': 'report_date'})

    del tr_clean_datapoints_df_stored
    clean.collect()

    # Merge exchange rate on country code and report date that datapoint comes from
    exc_rates_df['report_date'] = exc_rates_df['report_date'].astype(str)
    exch_df_resampled['report_date'] = exch_df_resampled['report_date'].astype(str)

    exc_rates_df = exc_rates_df.reset_index()
    exc_rates_df = exc_rates_df.dropna(subset = ['isocode', 'report_date'], how = 'any')
    exc_rates_df = exc_rates_df.set_index(['isocode', 'report_date'])
    
    exch_df_resampled = exch_df_resampled.reset_index()
    exch_df_resampled = exch_df_resampled.dropna(subset = ['isocode', 'report_date'], how = 'any')
    exch_df_resampled = exch_df_resampled.set_index(['isocode', 'report_date'])

    exc_rates_df = exc_rates_df.join(exch_df_resampled.drop(columns = ['file', 'filedate',
       'date_created']))

    del curr_df, exch_df_resampled
    clean.collect()

    # Return to asset/date index
    exc_rates_df = exc_rates_df.reset_index().set_index(['assetid', 'date'])

    exc_rates_df = exc_rates_df.sort_index()

    # Fill USD exchange rates with 1
    exc_rates_df['exch_rate_usd'].loc[exc_rates_df['isocode'] == 'USD'] = 1
    exc_rates_df['exch_rate_per_usd'].loc[exc_rates_df['isocode'] == 'USD'] = 1

    # Convert datapoints
    for tr_score in curr_con_dict.keys():
        tr_esg_df_norm[tr_score] = tr_esg_df_norm[tr_score].astype(float) * exc_rates_df['exch_rate_per_usd']

    del exc_rates_df
    clean.collect()

    # convert whole dataframe to numeric
    tr_esg_df_norm = tr_esg_df_norm.apply(pd.to_numeric)

    # reverse the polarity for identified variables
    for col in tr_esg_df_norm.columns:
        if polarity_dict[col][0] == 'Negative':
            tr_esg_df_norm[col] = -1 * tr_esg_df_norm[col]
    
    #join meta data
    tr_zscores = tr_esg_df_norm.join(meta)
    
    # convert scores to nans if company is not in correct sector 
    for score in sector_dict.keys():
        tr_zscores[score].loc[~(tr_zscores['industry'].isin(sector_dict[score]) +  tr_zscores['economic_sector'].isin(sector_dict[score]))] = np.nan
        
    tr_zscores = tr_zscores.drop(columns = ['industry', 'economic_sector'])

    # z-score then %rank 
    for col in tr_zscores.columns:
        tr_zscores[col] = tr_zscores[col].groupby(level='date').apply(lambda x: z_score(x))

    # convert z-scores to %s using the cumulative distribution function
    tr_scores_norm = pd.DataFrame(index=tr_zscores.index, columns=tr_zscores.columns,
                                  data=100 * stats.norm.cdf(tr_zscores))

    del tr_esg_df_norm, tr_zscores
    clean.collect()

    return tr_scores_norm
        
        

def load_revenue(start, end, timestamp):
    """
    Loads Factset financial data  and converts to USD
    Executes SQL statement using the following Tables & variables:
        * ``AraBaDa.dbo.FILTERED_TS_F_SALES_A``
            * assetid
            * date
            * filedate
            * clpyearend
            * all business involvement columns from input_dict
    Args:
        d_name: factset datapoint to be loaded
        country_data: dataframe of reporting currency per assetid
        currency_df: dataframe of USD conversion rates per currency
        start (Datetime): date (-2 years) to start filtering data
        end (Datetime): date to stop filtering data
        assets (tuple, defualt empty): if not empty, filters these assets out 
        for testing
    Returns:
        Pandas DataFrame: DataFrame containing  Factset financial data for the 
        datapoint d_name, converted into USD,data with multiIndex of 
        [assetid, date]
    """
    
    load_start = dt.strftime(pd.to_datetime(start) - pd.DateOffset(years=2),
                             "%Y-%m-%d")
    
    fs_sql = """select asset_id as assetid,
    viewtimeutc as date,
    data_column2 as revenue,
    data_column3 as report_date
    from filtered_ts_f_sales_a 
    where viewtimeutc >= '{0}' and  viewtimeutc <= '{1}'
                """.format(load_start, end)
                
    # Load asset meta data
    asset_entity = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)], as_of = timestamp)
    asset_sec = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], as_of = timestamp)
    
    asset_sec.index.names = ["assetid"]
    asset_entity.index.names = ["assetid"]
                
    client = DBClient(
    db_engine=psycopg2, config=db_config
    )

    conn = client.get_connection()

    cursor = conn.cursor()
    cursor.execute(fs_sql)
    conn.commit()
    
    fs_raw_df = pd.read_sql(fs_sql, conn)
    
    # Recreate original SQL quer (commented out above) by merging and selecting
    fs_raw_df_comb = fs_raw_df.join(asset_entity['primary_assetid'], 
                                    how = 'left')
    fs_raw_df_comb = fs_raw_df_comb.set_index(['assetid', 'date'])
    fs_raw_df_comb = fs_raw_df_comb.loc[fs_raw_df_comb.index.get_level_values(
            'assetid') == fs_raw_df_comb['primary_assetid']]
    
    fs_df = fs_raw_df_comb.join(asset_sec['economic_sector'], how = 'left')
    fs_df = fs_df.loc[fs_df['economic_sector'] != 'Government']
    
    # Drop unwanted columns
    fs_df = fs_df.drop(
            columns = ['primary_assetid', 'economic_sector']).reset_index()
    
    fs_df['date'] = fs_df['date'].apply(pd.to_datetime)
    fs_df['report_date'] = fs_df['report_date'].apply(pd.to_datetime)
    
    fs_df_clean = clean_fs_fin(fs_df)

    idx = pd.IndexSlice
    fs_df_clean_out = fs_df_clean.loc[idx[:, start:end] , :]
    
    return fs_df_clean_out


def get_exchange_rates(start, end):
    """
    Fetches active assets and cleans the data.
    Executes SQL statment using the following Tables & variables:
        * ``tempdb.dbo.cached_100001_obj_view_asset_volume_start_end_dates``:
            * assetid
            * first_date
            * last_date
    Returns:
        Pandas Dataframe: containing the assetid, first_date, last_date for 
        active assets.
    """
    
    exch_query = """SELECT a.*
   
    FROM fx_rates_usd a
        where a.date >= '{0}' and  a.date <= '{1}'
                """.format(start, end)
    
    client = DBClient(
    db_engine=psycopg2, config=db_config
    )

    conn = client.get_connection()
    
    cursor = conn.cursor()
    cursor.execute(exch_query)
    conn.commit()
    
    exch_df = pd.read_sql(exch_query, conn)
    
    exch_df = exch_df.rename(columns={'asset_id': 'assetid'})
    exch_df = exch_df.rename(columns={'iso_currency': 'isocode'})


    curr_query = """SELECT *
    FROM assetcurrencyinfo """ 
    
    cursor = conn.cursor()
    cursor.execute(curr_query)
    conn.commit()
    
    curr_df = pd.read_sql(curr_query, conn)
    
    curr_df = curr_df.rename(columns={'asset_id': 'assetid'})

    return exch_df, curr_df


def clean_fs_fin(fs_raw_df):
    """
    Cleans FactSet financial data by resampling, filling missing data and removing old data
    Method:
        * DF is cleaned and data older than 2 years is removed
        * Inputs with non-unique values are removed
        * Data is resampled to weekly
        * Data is forward filled up to 1 year
    Args:
        fs_df (Pandas Dataframe): Dataframe of raw factset data values
    Returns:
        Pandas Dataframe: Dataframe of cleaned Thomson Reuters data values
        :param normalize:
    """

    # Clean datatypes
    fs_clean_df = clean_df(fs_raw_df, ["assetid", "date"])
    # fs_clean_df.index.names = [PrimaryKey.assetid, PrimaryKey.date]
    del fs_raw_df
    clean.collect()

    # Drop columns that are all null
    fs_clean_df.dropna(how="all", axis=1, inplace=True)

    # Resample data to weekly frequency and until today and forward fill up to 2 years
    fs_df = resample(fs_clean_df, "D", "ffill", 365 * 2 + 1)
    del fs_clean_df
    clean.collect()

    return fs_df


def clean_df(df, index, drop_cols=True):
    """
    Optimises datatype memory footprint, drops empty columns, sets and sorts on given index.
    Args:
        df (Pandas DataFrame): input DataFrame
        index (str): DataFrame index
        drop_cols (bool, optional): columns to drop
    Returns:
        Pandas Dataframe: cleaned DataFrame
    """
    # Copy input dataframe
    df_clean = df.copy()
    del df
    clean.collect()

    # Optimize datatype memory footprint
    for col in df_clean.select_dtypes(include=["datetime"]).columns:
        df_clean[col] = pd.DatetimeIndex(df_clean[col]).normalize()

    for dtype in ["float", "integer"]:
        for col in df_clean.select_dtypes(include=[dtype]).columns:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast=dtype)
    clean.collect()

    # Drop empty columns
    if drop_cols:
        df_clean.dropna(how="all", axis=1, inplace=True)

    # Set and sort index
    if index is not None:
        df_clean.set_index(index, drop=True, inplace=True)
        df_clean.sort_index(inplace=True)

    if df_clean.empty:
        logger.info("WARNING: DataFrame is empty after cleaning.")
    else:
        length = df_clean.shape[0]

        logger.info("Success! DataFrame length {0} cleaned.".format(str(length)))

    return df_clean


def round_df(df, dig):
    """
    Rounds values in dataframe to 2 decimal places
    Args:
        df (Pandas Dataframe): Dataframe that needs rounding
        dif (unused)
    Returns:
        Pandas Dataframe: dataframe containing rounded values
        :param dig:
    """
    new_df = df.copy()
    for col in new_df:
        new_df[col] = new_df[col].apply(lambda x: round(x, dig))

    del df
    clean.collect()

    if new_df.empty:
        logger.info("WARNING: DataFrame is empty after rounding.")
    else:
        date_max = new_df.index.get_level_values('date').max()
        idx = pd.IndexSlice
        count = len(new_df.loc[idx[:, date_max], :].index.get_level_values('assetid').unique())

        logger.info("Success! DataFrame rounded: {0} assets as of {1}.".format(str(count), str(date_max.date())))

    return new_df




def clean_dict(dict_raw, idx_name, col_name, avl_inputs=None):
    """
    Cleans dictionary inputs by removing non-available inputs
    Args:
        dict_raw (dict): raw dictionary to be cleaned
        idx_name (str): name for key index
        col_name (str): name of values/columns
        avl_inputs (list, optional): sub_set of values to filter from raw dictionary
    Returns:
        dict: dict of available inputs where Keys: idx_name
                                             Values: column names
    """
    # Remove columns which are not available from the loaded data
    if avl_inputs:
        dict_avl = dict_raw[dict_raw[col_name].isin(avl_inputs)]
    else:
        dict_avl = dict_raw.copy()

    # Clean datatypes and set/sort index
    dict_clean = clean_df(dict_avl, idx_name)

    # Construct empty dictionary
    dict_ = {}

    # Fill up dictionary with key-value pairs
    for col in np.unique(list(dict_clean.index)):
        if len(dict_clean.loc[col, [col_name]]) == 1:
            dict_[col] = dict_clean.loc[col, [col_name]].tolist()
        elif len(dict_clean.loc[col, [col_name]]) > 1:
            dict_[col] = list(set(dict_clean.loc[col, col_name].tolist()))
    del dict_clean
    clean.collect()

    if not dict_:
        logger.info("WARNING: Dict is empty after cleaning.")
    else:
        count = len(dict_.keys())
        logger.info("Success! Dictionary of {0} features cleaned.".format(str(count)))

    return dict_



# Resample data
def resample(df, freq, method, limit, id_idx='assetid',
             date_idx='date'):
    """
    Resamples data to the given frequency and fills in missing data up to the provided limit
    Args:
        df (Pandas DataFrame): input DataFrame.
        freq (str): the offset string representing target conversion for resampling.
        method (str): method to use for filling missing data.
        limit (int): this is the maximum number of consecutive NaN values to forward/backward fill with data
        drop_rows (bool, optional): If True, remove empty rows.
        drop_inactive (bool, optional): If True, removes observations for non-active listings
        idx_name (str, optional): Name of asset index in Dataframe.
        date_idx (str, optional): NAme of date index in Dataframe.
    Returns:
        Pandas Dataframe: resampled Dataframe
        :param id_idx:
    """
    # Copy input dataframe
    df_raw = df.copy()
    del df
    clean.collect()

    # Get original list of columns
    cols = list(df_raw.columns)

    # Make sure the dataframe is sorted on the index
    df_raw.sort_index(level=[id_idx, date_idx], inplace=True)

    # Add latest date to dataframe
    today = pd.to_datetime(dt.now()).normalize()
    lr_df = df_raw.groupby(level=id_idx).tail(1).copy()
    lr_df.reset_index(date_idx, drop=False, inplace=True)
    lr_df.loc[:, date_idx] = today
    lr_df.set_index(date_idx, append=True, drop=True, inplace=True)
    lr_df.loc[:, cols] = np.nan
    lr_df = lr_df.reorder_levels(df_raw.index.names)
    df_last = df_raw.append(lr_df).sort_index(level=[id_idx, date_idx])

    # Resample dataframe and merge back dropped indices
    df_resampled = df_last.groupby(level=id_idx).resample(freq, level=date_idx).last()

    # Avoid future observations
    if len(df_resampled) > 0:
        idx = pd.IndexSlice
        date_max = idx[:, :today]
        df_resampled = df_resampled.loc[date_max, :]

    # Feed forward missing values
    if method is not None:
        df_resampled = df_resampled.groupby(level=id_idx).fillna(method=method, limit=limit)

    return df_resampled


def z_score(x):
    if x.std(ddof=0) == 0:
        return x * 0
    else:
        return (x - x.mean()) / x.std(ddof=0)