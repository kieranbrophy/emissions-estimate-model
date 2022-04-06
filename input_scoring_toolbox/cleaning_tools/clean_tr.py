#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:15:13 2021

@author: roandufeu
"""

# Loading in libraries
import pandas as pd
import gc as clean

from input_scoring_toolbox.cleaning_tools import cleaning_tools as cf
import logging
logger = logging.getLogger(__name__)


# TR  cleaning functions
def clean_tr_datapoints(tr_datapoints_raw_df, start, end, frequency ='D' , make_primary = True):
    """
    Cleans Thomson Reuters data by resampling, filling missing data and removing old data
    Method:
        * DF is cleaned and data older than 2 years is removed
        * Inputs with non-unique values are removed
        * Data is resampled to weekly
        * Data is forward filled up to 1 year
    Args:
        tr_datapoints_df (Pandas Dataframe): Dataframe of raw TR datapoints
        start (Datetime): start date of data
        end (Datetime): end date of data
        frequency (string): frequency of cleaned data
        make_primary (bool): convert all assetids to primary_assetids
    Returns:
        Pandas Dataframe: Dataframe of cleaned Thomson Reuters data values
        :param normalize:
    """
    
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    tr_datapoints_raw_df['date'] = tr_datapoints_raw_df['date'].apply(pd.to_datetime)
    tr_datapoints_raw_df['clpyearend'] = tr_datapoints_raw_df['clpyearend'].apply(pd.to_datetime)
    tr_datapoints_raw_df['filedate'] = tr_datapoints_raw_df['filedate'].apply(pd.to_datetime)
        
    # Clean datatypes
    tr_datapoints_clean_df = cf.clean_df(tr_datapoints_raw_df, None, drop_cols = False)
    del tr_datapoints_raw_df
    clean.collect()

    # Drop obs > 2 years old and rename
    tr_datapoints_clean_df["date_diff"] = (tr_datapoints_clean_df["date"] - tr_datapoints_clean_df["clpyearend"]).apply(lambda x: x.days / 365)
    tr_datapoints_clean_df = tr_datapoints_clean_df[tr_datapoints_clean_df["date_diff"] <= 2]
    tr_datapoints_clean_df.drop(["filedate", "date_diff"],axis=1, inplace=True)
    tr_datapoints_clean_df.rename(columns={"clpyearend": "report_date"}, inplace=True)


    # Resample data to weekly frequency and until today
    tr_datapoints_df = cf.clean_resample(tr_datapoints_clean_df, "W", None, None, drop_rows=False, drop_cols=False, drop_inactive=False, id_idx="assetid",
             date_idx="date", until=end, make_primary = make_primary)
    del tr_datapoints_clean_df
    clean.collect()

    idx = pd.IndexSlice
    historical = idx[:, :"2020-01-25"] # Data for all complete history - arrived on 2020-01-25
    live = idx[:, "2020-01-19":] # Live period (historical - 6 days to account for weekly resampling)

    tr_datapoints_df.loc[historical, :] = tr_datapoints_df.loc[historical,:].groupby(level="assetid").fillna(method="ffill", limit=52*2+1)
    tr_datapoints_df.loc[live, :] = tr_datapoints_df.loc[live, :].groupby(level="assetid").fillna(method="ffill", limit=8)

    # Remove empty rows and overly outdated data
    tr_datapoints_df.dropna(how="all", axis=0, inplace=True)
    tr_datapoints_df["obsdate"] = tr_datapoints_df.index.get_level_values("date")
    tr_datapoints_df["age"] = (tr_datapoints_df["obsdate"] - tr_datapoints_df["report_date"]).apply(lambda x: x.days / 365)
    tr_datapoints_df = tr_datapoints_df[tr_datapoints_df["age"] <= 2].copy()
    tr_datapoints_df.drop(["obsdate", "age"], axis=1, inplace=True)

    # Remove non-active listings
    tr_datapoints_df = cf.remove_inactive(tr_datapoints_df, 'date')
    
    # select data starting a week before the calculation period 
    start_calc = pd.to_datetime(start) - pd.DateOffset(days=6)
    end_calc = pd.to_datetime(end) + pd.DateOffset(days=1)

    calc_period = pd.IndexSlice[:, start_calc:end_calc]
    tr_datapoints_df_calc = tr_datapoints_df.loc[calc_period, :]
    
    del tr_datapoints_df
    clean.collect()
    
    if frequency == 'D':
        fill_limit = 6
    else: 
        fill_limit = 1
    
    # Resample to daily and forward fill by a week
    tr_datapoints_df_out = cf.clean_resample(tr_datapoints_df_calc.reset_index(), frequency, 'ffill', fill_limit, drop_rows=False, drop_cols=False, 
                                             drop_inactive=False, id_idx="assetid", date_idx="date", until=end, make_primary = False)
    
    del tr_datapoints_df_calc
    clean.collect()
    
    # Take only data from the calculation period
    tr_datapoints_df_out = tr_datapoints_df_out.loc[pd.IndexSlice[:, start:end] , :]
    
    # Remove null entries
    tr_datapoints_df_out = tr_datapoints_df_out.dropna(how = 'all')

    # replace col names with trd_
    cols = [col.replace('tr_', 'trd_') for col in tr_datapoints_df_out.columns]
        
    tr_datapoints_df_out.columns = cols
    
    if tr_datapoints_df_out.empty:
        logger.info("WARNING: No data after cleaning for TR datapoints.")
    else:
        count = len(tr_datapoints_df_out.index.get_level_values('assetid').unique())
        
        logger.info(f"Success! TR datapoints cleaned: {count} assets in {len(tr_datapoints_df_out)} rows.")
    
    return tr_datapoints_df_out


# TR  cleaning functions
def clean_tr_values(tr_values_raw_df, start, end, frequency ='D', make_primary = True):
    """
    Cleans Thomson Reuters data by resampling, filling missing data and removing old data
    Method:
        * DF is cleaned and data older than 2 years is removed
        * Inputs with non-unique values are removed
        * Data is resampled to weekly
        * Data is forward filled up to 1 year
    Args:
        tr_datapoints_df (Pandas Dataframe): Dataframe of raw TR datapoints
        start (Datetime): start date of data
        end (Datetime): end date of data
        frequency (string): frequency of cleaned data
        make_primary (bool): convert all assetids to primary_assetids
    Returns:
        Pandas Dataframe: Dataframe of cleaned Thomson Reuters data values
        :param normalize:
    """
    
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    
    tr_values_raw_df['date'] = tr_values_raw_df['date'].apply(pd.to_datetime)
    tr_values_raw_df['clpyearend'] = tr_values_raw_df['clpyearend'].apply(pd.to_datetime)
    tr_values_raw_df['filedate'] = tr_values_raw_df['filedate'].apply(pd.to_datetime)

    # Clean datatypes
    tr_values_clean_df = cf.clean_df(tr_values_raw_df, None, drop_cols = False)
    del tr_values_raw_df
    clean.collect()

    # Drop obs > 2 years old and rename
    tr_values_clean_df["date_diff"] = (tr_values_clean_df["date"] - tr_values_clean_df["clpyearend"]).apply(lambda x: x.days / 365)
    tr_values_clean_df = tr_values_clean_df[tr_values_clean_df["date_diff"] <= 2]
    tr_values_clean_df.drop(["filedate", "date_diff"],axis=1, inplace=True)
    tr_values_clean_df.rename(columns={"clpyearend": "report_date"}, inplace=True)

    # Resample data to weekly frequency and until today
    tr_values_df = cf.clean_resample(tr_values_clean_df, "W", None, None, drop_rows=False, drop_cols=False, drop_inactive=False, id_idx="assetid",
             date_idx="date", until=end, make_primary = make_primary)
    del tr_values_clean_df
    clean.collect()

    idx = pd.IndexSlice
    historical = idx[:, :"2017-10-04"] # Data for all complete history - loaded on 2017-10-04
    live_1 = idx[:, "2017-09-28":"2019-08-28"]
    live_2 = idx[:, "2019-08-22":"2020-01-15"] # no data delivery after 2019-08-29 until 2020-01-16
    live_3 = idx[:, "2020-01-09":] # Live period

    tr_values_df.loc[historical, :] = tr_values_df.loc[historical,:]\
        .groupby(level="assetid").fillna(method="ffill", limit=52*2+1)
    tr_values_df.loc[live_1, :] = tr_values_df.loc[live_1,:]\
        .groupby(level="assetid").fillna(method="ffill", limit=8)
    tr_values_df.loc[live_2, :] = tr_values_df.loc[live_2, :]\
        .groupby(level="assetid").fillna(method="ffill", limit=20)
    tr_values_df.loc[live_3, :] = tr_values_df.loc[live_3, :]\
        .groupby(level="assetid").fillna(method="ffill", limit=8)

    # Remove empty rows and overly outdated data
    tr_values_df.dropna(how="all", axis=0, inplace=True)
    tr_values_df["obsdate"] = tr_values_df.index.get_level_values("date")
    tr_values_df["age"] = (tr_values_df["obsdate"] - tr_values_df["report_date"]).apply(lambda x: x.days / 365)
    tr_values_df = tr_values_df[tr_values_df["age"] <= 2].copy()
    tr_values_df.drop(["obsdate", "age"], axis=1, inplace=True)

    # Remove non-active listings
    tr_values_df = cf.remove_inactive(tr_values_df, 'date')

    # select data starting a week before the calculation period 
    start_calc = pd.to_datetime(start) - pd.DateOffset(days=6)
    end_calc = pd.to_datetime(end) + pd.DateOffset(days=1)

    calc_period = pd.IndexSlice[:, start_calc:end_calc]
    tr_values_df_calc = tr_values_df.loc[calc_period, :]
    
    del tr_values_df
    clean.collect()

    if frequency == 'D':
        fill_limit = 6
    else: 
        fill_limit = 1
    
    # Resample to daily and forward fill by a week
    # tr_values_df_out = cf.resample(tr_values_df_calc, 'D', 'ffill', 6, until=end)
    tr_values_df_out = cf.clean_resample(tr_values_df_calc.reset_index(), frequency, 'ffill', fill_limit, drop_rows=False, drop_cols=False, 
                                             drop_inactive=False, id_idx="assetid", date_idx="date", until=end, make_primary = False)
    
    
    del tr_values_df_calc
    clean.collect()
    
    # Take only data from the calculation period
    tr_values_df_out = tr_values_df_out.loc[pd.IndexSlice[:, start:end] , :]
    
    # Remove null entries
    tr_values_df_out = tr_values_df_out.dropna(how = 'all')

    # replace col names with trd_
    cols = [col.replace('tr_', 'trd_') for col in tr_values_df_out.columns]
        
    tr_values_df_out.columns = cols

    if tr_values_df_out.empty:
        logger.info("WARNING: No data after cleaning for TR values.")
    else:
        count = len(tr_values_df_out.index.get_level_values('assetid').unique())
        
        logger.info(f"Success! TR values cleaned: {count} assets in {len(tr_values_df_out)} rows.")
    
    return tr_values_df_out