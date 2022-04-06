#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:58:40 2021

@author: roandufeu
"""

import gc as clean
import psycopg2
import input_scoring_toolbox.loading_tools as lt
import input_scoring_toolbox.cleaning_tools as ct
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from scipy.sparse.csgraph._traversal import connected_components

from input_scoring_toolbox.loading_tools.db_client.client import DBClient
from input_scoring_toolbox.loading_tools.db_loader.loader import DBLoader
from input_scoring_toolbox.config import db_config


def clean_fs_inv(raw_data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, 
                 freq:str = 'D', field_names: list = None, assets: list = None, 
                 fill_nans:bool = True,  drop_inactive:bool = True) -> pd.DataFrame:

    """
    Cleans Factset Hierarchy data by pivoting, resampling and filling missing data

    Method:
        * DF is pivoted so selected revenue streams become columns
        * involvements entirely before start_dt are removed
        * start dates before load_start are updated to load_start
        * start dates are labelled 1 and end dates 0
        * data is resampled to weekly
        * data is forward filled up to today

    Parameters
    ----------
        raw_data: raw FS data values
        end: date to stop filtering data

    Returns
    -------
        Pandas Dataframe: Dataframe of cleaned FS hierarchy data
    """
        
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    load_start = pd.to_datetime(start) - pd.DateOffset(weeks=1)
    
    
    
    # pull universe of assets 
    if assets:
        universe = lt.get_raw_data('fs_inv_universe_noview',  load_start, end, entity_list = assets)
    else:
        universe = lt.get_raw_data('fs_inv_universe_noview',  load_start, end)
    
    
    # if involvement data for an asset in a sector is present as several (more than 2) records
    # (record 1: start date A - end date B, record 2: start date B - end date C),
    # merge those records into one (start date A - end date C).

    logger.debug('Finding records to merge')
    grouped = raw_data.groupby(['assetid', 'sector_id'])
    grouped_size = grouped.size()
    to_merge_list = list(grouped_size[grouped_size >= 2].index)

    logger.debug('Seperating out records that may need to be merged')
    raw_data['asset_sector'] = list(zip(raw_data.assetid, raw_data.sector_id))
    to_merge = raw_data[raw_data.asset_sector.isin(to_merge_list)].drop(columns='asset_sector')
    raw_data = raw_data[~raw_data.asset_sector.isin(to_merge_list)].drop(columns='asset_sector')

    logger.debug('Merging & recombining rows with the same involvement for overlapping time periods')
    to_merge = to_merge.groupby(['assetid', 'sector_id']).apply(merge_rows)
    raw_data = pd.concat([raw_data, to_merge])

    # add one day to end_date (this ensures when forward filling later, that
    #   'involvement' is filled forward from start date to end date, including end date)
    raw_data['end_date'] = raw_data['end_date'] + pd.DateOffset(days=1)

    # Set irrelevant sector ids to -1.
    # We want to preserve the index, so can't throw them away yet.
    # field_sector_ids = [str(sector_name.strip("hier_")) for sector_name in field_names]
    # raw_data['sector_id'] = raw_data['sector_id'].apply(lambda sid: sid if (str(sid) in field_names) else '-1')

    # update all start dates before load_start to load_start
    # load_start = start - pd.DateOffset(weeks=1)
    raw_data['date'].loc[raw_data['date'] <= load_start] = load_start
    raw_data = raw_data.set_index(["assetid", "date", "end_date", "sector_id"])

    # Throw away duplicates for irrelevant sectors
    raw_data = raw_data.loc[~raw_data.index.duplicated(keep='first')]

    # make factset row data into columns
    logger.debug('Unstacking sector_id column to create new column names')
    fs_hier_df = raw_data.unstack(level="sector_id")
    fs_hier_df.columns = fs_hier_df.columns.droplevel()

    del to_merge, raw_data
    clean.collect()

    # Data at this point has both start date (date) and end date (end_date) for involvement
    # as columns in the index. We want a single date column with 1s for involvement
    # and 0s for non-involvement so we can forward fill from start date to end date

    # We need to split the dataframe in two, set 'starts' as 1 and 'ends' as 0, then recombine and sort.
    # After doing this we can forward fill, resulting in 1 for involvement and 0 for non-involvement for each date

    # split into separate dataframes of start dates and end dates
    fs_hier_df_start_dates = fs_hier_df.reset_index("end_date", drop=True)
    fs_hier_df_end_dates = fs_hier_df.reset_index("date", drop=True)

    fs_hier_df_end_dates.index.rename('date', level=1, inplace=True)
    # change end date values to 0's to signify non involvement in the sector
    fs_hier_df_end_dates.replace({1: 0}, inplace=True)
    # concatenate the two dataframes so for each sector involvement start & end dates
    # are signified in the same column with 1s & 0s respectively
    fs_hier_date_df = pd.concat([fs_hier_df_start_dates, fs_hier_df_end_dates])
    fs_hier_date_df = fs_hier_date_df.sort_index()

    del fs_hier_df, fs_hier_df_end_dates, fs_hier_df_start_dates
    clean.collect()
    # combine multiple start/ end dates together - if a date is marked as both a start and an end date
    # (could happen if information in a row was ammended but involment did not stop) then take the last value
    # (maintaining consistency with resampling which is also based on last value)
    logger.debug('Merging repeated dates')
    fs_hier_date_df = fs_hier_date_df.groupby(level=[0, 1]).last()

    # clean the dataframe
    fs_clean_df = ct.clean_df(fs_hier_date_df, None, drop_cols = False)
    logger.debug(f"Resampling {fs_clean_df.shape}")

    fs_out_df= ct.clean_resample(fs_clean_df.reset_index(), freq, "ffill", None, drop_rows = False, drop_cols = False, 
                                  drop_inactive = True, until=(end  + pd.DateOffset(weeks=1)), make_primary = True)
    
    fs_out_df = fs_out_df.loc[fs_out_df.index.get_level_values('date') > (start - pd.DateOffset(weeks=1))]
    
    del fs_clean_df, fs_hier_date_df
    clean.collect()
    
    idx = pd.IndexSlice
    time_period = idx[:, start:end]
    fs_out_df = fs_out_df.loc[time_period, :] 
    
    # fill in final dataframe with all the app fields from fs_inv_cleaned_daily
    
    if not field_names:
        field_names = fs_out_df.columns
    
    fs_out_all = pd.DataFrame(index = pd.MultiIndex.from_product([universe.set_index('assetid').index,  
                                                                  pd.date_range(start, end, freq = freq)  ],
                                                                 names = ['assetid', 'date']),
                              columns = field_names,
                              data = fs_out_df)
    
    # if new columns are appended from app fields values, replace nans with zeros - no involvement in this sector
    if fill_nans:
        fs_out_all = fs_out_all.fillna(0)
    
    # Remove observations for non-active listings
    logger.debug('Removing non-active listings.')
    
    if drop_inactive:
        fs_out_all = ct.remove_inactive(fs_out_all, 'date')

    fs_out_all = fs_out_all.add_prefix("inv_")

    return fs_out_all


def clean_fs_rev(raw_data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, 
                 freq:str = 'D', field_names: list = None, assets: list = None, 
                 fill_nans:bool = True, drop_inactive:bool = True) -> pd.DataFrame:
    """
    Cleans Factset segrev data by pivoting, resampling and filling missing data

    Method:
        * DF is pivoted so selected revenue streams become columns
        * Only assets with reports are used - and the date of the whole report is used
            - note this will be the most recent date of all the records associated with that report
        * Data is resampled to weekly
        * Data is forward filled up to 'ffill' years

    Parameters list(fs_rev_universe['assetid'])
    ----------
        raw_data: dataframe of raw FS segrev data values
        
        end: date to stop filtering data

    Returns
    -------
        dataframe of cleaned FS segrev data values
    """
    
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    load_start = pd.to_datetime(start) - pd.DateOffset(years=2)
    
    
    # pull universe of assets 
    if assets:
        universe = lt.get_raw_data('fs_rev_universe_noview',  load_start, end, entity_list = assets)
    else:
        universe = lt.get_raw_data('fs_rev_universe_noview',  load_start, end)
    

    raw_data = raw_data.set_index(["assetid", "date", "sector_id"]).drop(columns=["report_id"])
    

    # make FactSet row data into columns
    logger.debug("Unstacking sector_id column to create new column names")
    fs_clean_df = raw_data.unstack(level="sector_id")
    fs_clean_df.columns = fs_clean_df.columns.droplevel()

    del raw_data
    clean.collect()

    # drop non-relevant sectors (sectors to be cleaned are defined by the app yaml)
    # logger.debug("Dropping non-relevant sectors from columns")
    # field_sector_ids = [sector_name.strip("rev_") for sector_name in field_names]
    # sector_ids = [_id for _id in field_sector_ids if _id in fs_clean_df.columns]
    # fs_clean_df = fs_clean_df[sector_ids]

    # if a row exists at this point it is because a report exists, so a nan here means zero income
    #   therefore, to avoid forward filling through a point where there is supposed to be a zero,
    #   at this point we can replace nans with zeros
    logger.debug("Filling missing revenue with zeros")
    fs_clean_df = fs_clean_df.fillna(0)

    fs_clean_df = ct.clean_df(fs_clean_df, None, drop_cols = False)
    # resample data to weekly frequency and until today
        
    resample_dict = {'D':731, 'W':105, 'M': 25, 'A':2 }
    fs_out_day= ct.clean_resample(fs_clean_df.reset_index(), freq = freq, method = "ffill", limit = resample_dict[freq], drop_rows = False, drop_cols = False, 
                                  drop_inactive = True, until=(end  + pd.DateOffset(weeks=1)), make_primary = True)
    
    fs_out_day = fs_out_day.loc[fs_out_day.index.get_level_values('date') > (start - pd.DateOffset(weeks=1))]
    
    del fs_clean_df
    clean.collect()
    
    idx = pd.IndexSlice
    time_period = idx[:, start:end]
    fs_out_day = fs_out_day.loc[time_period, :] 
    
    if not field_names:
        field_names = fs_out_day.columns
    
    logger.debug("Creating final rev dataframe")
    fs_out_all = pd.DataFrame(index = pd.MultiIndex.from_product([universe.set_index('assetid').index,  
                                                                  pd.date_range(start, end, freq = freq)  ],
                                                                 names = ['assetid', 'date']),
                              columns = field_names,
                              data = fs_out_day)
    
    if fill_nans:
        fs_out_all = fs_out_all.fillna(0)
    
    # occasionally factset data will report revenue percentage above 100 (due to companies selling to themselves)
    # clip to 100 max
    fs_out_all = fs_out_all.clip(0, 100)

    # Remove observations for non-active listings
    logger.debug('Removing non-active listings.')
    
    if drop_inactive:
        fs_out_all = ct.remove_inactive(fs_out_all, 'date')
    
    fs_out_all = fs_out_all.add_prefix("rev_")
 
    return fs_out_all


def merge_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges records into one, if date ranges overlap.
    """

    # create a 2D graph of connectivity between date ranges
    start = data.date.values
    end = data.end_date.values

    # connect records with overlapping date range
    graph = (start <= end[:, None]) & (end >= start[:, None])

    # find connected components in this graph
    n_components, indices = connected_components(graph)

    # group the results by these connected components
    return data.groupby(indices).aggregate({'date': 'min', 'end_date': 'max', 'assetid': 'first',
                                            'sector_id': 'first', 'flag': 'first'})



