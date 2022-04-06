#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:15:43 2021

@author: roandufeu
"""


# Loading in libraries
import pandas as pd
import gc as clean
import numpy as np
import logging
import psycopg2
logger = logging.getLogger(__name__)


import input_scoring_toolbox.loading_tools as lt
import input_scoring_toolbox.cleaning_tools as ct
from sray_db.apps import apps
from sray_db.apps.pk import PrimaryKey
from input_scoring_toolbox.config import db_config

from input_scoring_toolbox.loading_tools.db_client.client import DBClient


def clean_resample(df:pd.DataFrame, freq:str, method:str, limit:int, id_idx:str="assetid", date_idx:str="date", from_date: pd.Timestamp = None, until: pd.Timestamp = None, 
                   drop_rows:bool=True, drop_cols:bool=False, drop_inactive:bool=True, make_primary:bool = True):
    """
    Cleans and resamples data to the given frequency and fills in missing data up to the provided limit

    Args:
        df (Pandas DataFrame): input DataFrame.
        freq (str): the offset string representing target conversion for resampling.
        method (str): method to use for filling missing data.
        limit (int): this is the maximum number of consecutive NaN values to forward/backward fill with data
        drop_rows (bool, optional): If True, remove empty rows.
        drop_cols (bool, optional): If True, remove empty columns.
        drop_inactive (bool, optional): If True, removes observations for non-active listings
        id_idx (str, optional): Name of asset index in Dataframe.
        date_idx (str, optional): NAme of date index in Dataframe.
        until (pd.Timestamp, optional): maximum date to resample to
        from_date (pd.Timestamp, optional): minimum date to return

    Returns:
        Pandas Dataframe: resampled Dataframe

    """
    until = pd.Timestamp(until)
    
    logger.debug(f'Preparing to resample {len(df)} rows into frequency {freq}')
    # Copy input dataframe
    df_raw = df.copy()
    del df
    clean.collect()
    
    df_raw = df_raw.reset_index()
    if 'index' in df_raw.columns:
        df_raw = df_raw.drop(columns = 'index')
    
    if make_primary:
        df_raw = make_assets_primary(df_raw)
        
       
    df_raw = clean_df(df_raw, [id_idx, date_idx], drop_cols=False)

    # Get original list of columns
    cols = list(df_raw.columns)

    # Make sure the dataframe is sorted on the index
    logger.debug('Sorting index.')
    df_raw.sort_index(level=[id_idx, date_idx], inplace=True)

    # Add latest date to dataframe
    if until is not None:
        logger.debug(f'Adding last date {until}')
        lr_df = df_raw.groupby(level=id_idx).tail(1).copy()
        lr_df.reset_index(date_idx, drop=False, inplace=True)
        lr_df.loc[:, date_idx] = until
        lr_df.set_index(date_idx, append=True, drop=True, inplace=True)
        lr_df.loc[:, cols] = np.nan
        lr_df = lr_df.reorder_levels(df_raw.index.names)
        df_raw = df_raw.append(lr_df).sort_index(level=[id_idx, date_idx])

    # Resample dataframe and merge back dropped indices
    logger.debug('Resampling data.')
    df_resampled = df_raw.groupby(level=id_idx).resample(freq, level=date_idx).last()

    # Avoid future observations
    if len(df_resampled) > 0 and until:
        idx = pd.IndexSlice
        date_max = idx[:, :until]
        df_resampled = df_resampled.loc[date_max, :]

    # Feed forward missing values
    if method is not None:
        logger.debug('Feeding forward missing values')
        df_resampled = df_resampled.groupby(level=id_idx).fillna(method=method,limit=limit)
        
    if from_date:
        idx = pd.IndexSlice
        date_min = idx[:, from_date:]
        df_resampled = df_resampled.loc[date_min, :]

    # Remove empty rows
    if drop_rows:
        df_resampled.dropna(how="all",axis=0,inplace=True)
        
    # Remove observations for non-active listings
    if drop_inactive and len(df_resampled) > 0:
        logger.debug('Removing non-active listings.')
        df_resampled = remove_inactive(df_resampled, date_idx)
        
    logger.debug(f'Data resampled into {len(df_resampled)} rows.')
    return df_resampled

def metric_sector_applicability(data_df: pd.DataFrame, id_idx: str):
    """
    Automatically assigns companies nans if they are not in the correct sectors for a metric

    Args:
        data_df (Pandas DataFrame): input DataFrame.
        id_idx (str): company identifier (assetid or entity_id)

    Returns:
        Pandas Dataframe: filtered Dataframe

    """
    
    sharepoint_fetcher = lt.SPLoader()
    
    metadata_df = sharepoint_fetcher.get_excel('Data/input_scoring_toolbox_metadata.xlsx', sheet_name = 'input_scoring_toolbox_metadata')[['S-Ray Key', 'Sector Applicability', 'Revere Applicability']]
    meta_sector_df = metadata_df.loc[ ~metadata_df['Sector Applicability'].isnull()]
    
    meta_sector_df['Sector Applicability'] = meta_sector_df[['Sector Applicability']].apply(lambda x: x.str.split(','))
    
    sector_dict = meta_sector_df.set_index('S-Ray Key').to_dict()
    sector_dict = sector_dict['Sector Applicability']
    
    sector_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)])
    sector_df.index.names = ['assetid']
    
    if id_idx=='entity_id':
        sector_df = assetid_to_entityid(sector_df).drop_duplicates().set_index('entity_id')        
        
    df_filtered = data_df.join(sector_df, how = 'left', on = id_idx)
    
    for metric in data_df.columns:
        if metric in sector_dict.keys():
            df_filtered[metric].loc[~df_filtered['economic_sector'].isin(sector_dict[metric])] = np.nan
            
    df_filtered = df_filtered.drop(columns = ['economic_sector', 'industry'])
    
    return df_filtered


def fill_nans_all(data_df: pd.DataFrame, metric_list: list = None):
    """
    Fill nans with 0s for metrics specified by a metric list. If no metric list 
    provided, metrics specified in Data/input_scoring_toolbox_metadata.xlsx are filled

    Args:
        data_df (pd.DataFrame): dataframe to fill nans in
        metric_list (str): list of metrics to zero fill

    Returns:
    -------
    pd.DataFrame: dataframe with nans filled for specified metrics

    """
    
    sharepoint_fetcher = lt.SPLoader()
    
    if not metric_list:
        metadata_df = sharepoint_fetcher.get_excel('Data/input_scoring_toolbox_metadata.xlsx', sheet_name = 'input_scoring_toolbox_metadata')[['S-Ray Key', 'Fill Zeros']]
        meta_zeros_df = metadata_df.loc[ ~metadata_df['Fill Zeros'].isnull()]
        
        metric_list = meta_zeros_df['S-Ray Key'].to_list()
    
    for metric in data_df.columns:
        if metric in metric_list:
            data_df[metric] = data_df[metric].fillna(0)
    
    return data_df


def fill_nans_revere_sectors(data_df: pd.DataFrame, id_idx: str, start: str, load_start: str, end:str , freq:str):
    """
    Automatically fills nan values with zeros if the company has reported revenue
    in the relevant sectors for a metric

    Args:
        data_df (Pandas DataFrame): input DataFrame.
        id_idx (str): company identifier (assetid or entity_id)
        start (str): start date of scoring period
        end (str): end date of scoring period
        load_start (str): start date of lookbak period (usually 2 years before 'start')
        

    Returns:
        Pandas Dataframe: filtered Dataframe

    """
    
    sharepoint_fetcher = lt.SPLoader()
    
    metadata_df = sharepoint_fetcher.get_excel('Data/input_scoring_toolbox_metadata.xlsx', sheet_name = 'input_scoring_toolbox_metadata')[['S-Ray Key', 'Revere Applicability']]
    meta_revere_df = metadata_df.loc[ ~metadata_df['Revere Applicability'].isnull()]
    
    meta_revere_df['Revere Applicability'] = meta_revere_df[['Revere Applicability']].apply(lambda x: x.str.split(','))
    
    revere_dict = meta_revere_df.set_index('S-Ray Key').to_dict()
    revere_dict = revere_dict['Revere Applicability']
    
    
    if id_idx=='entity_id':
        assets = entityid_to_assetid(data_df, keep_ids = True, drop_unmatched=False).reset_index()['assetid']
        assets = list(assets.dropna().unique().astype(int).astype(str))
    else:
        assets = data_df.reset_index()['assetid']
        assets = list(assets.dropna().unique().astype(int).astype(str))
    
    # load fs revere data
    if revere_dict:
        # get fs sectors to load
        fs_sectors = []
        for value in revere_dict.values():
            fs_sectors = fs_sectors + value
        fs_sectors = list(set(fs_sectors))
        fs_sector_ids = [sector.strip('rev_') for sector in fs_sectors]
        
    revere_df = lt.get_raw_data('fs_rev', start = load_start, end = end, 
                                entity_list = assets, fs_load_sectors=fs_sector_ids)  
    
    if revere_df.empty:
        logger.info("INFO: No companies have revere data in relevant sectors.")
        return data_df
    revere_df_clean = ct.clean_fs_rev(revere_df, start, end, freq = freq, 
                                      assets=assets, field_names=fs_sector_ids)
        
        
    if id_idx=='entity_id':
        revere_df_clean = assetid_to_entityid(revere_df_clean).drop_duplicates().set_index(['entity_id', 'date'])  
    
    df_filtered = data_df.copy()
    
    for metric in data_df.columns:
        if metric in revere_dict.keys():
            columns = revere_dict[metric]
            df_filtered[metric].loc[ (revere_df_clean[columns].max(axis=1)>0) &( df_filtered[metric].isna() )] =  0
    
    return df_filtered

def convert_identifier(data_df: pd.DataFrame, id_from:str, id_to:str, keep_id_from:bool = False,
                       drop_unmatched:bool = True,  match_to_primary:bool = False, drop_duplicates:bool = True):
    """
    Converts a dataframe indexed on an certain identifier to a different identifier. Options are
    'assetid', 'primary_assetid', 'entity_id', 'isin', 'fspermid', 'fsymid',
       'sedol', 'ticker_exchange', 'ticker_region',  'cusip'

    Args:
        data_df (Pandas DataFrame): input DataFrame.
        id_from (str): current company identifier
        id_to (str): company identifier to map to
        keep_from_ids (bool): keep assetid as a seperate column
        drop_unmatched (bool): drop rows that do not map to an entity_id
        match_to_primary (bool): convert to primary assetid

    Returns:
        Pandas Dataframe: Dataframe with entity_id

    """
    # get final index
    index_names = list(data_df.index.names)
    index_names.remove(id_from)
    index_names.insert(0, id_to)
    
    # drop nans in incoming id_from
    data_df = data_df[~data_df.index.get_level_values(id_from).isnull()]
    
    if id_to == 'assetid' and match_to_primary == True:
        id_to = 'primary_assetid'
    
    # load entity_ids
    entity_ids = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)])
    entity_ids.index.names = ['assetid']
    entity_ids = entity_ids.rename(columns = {'fsentityid': 'entity_id'})
    
    
    # load security_ids
    security_ids = lt.get_meta_data(apps['assetinfo_security_id'][(1,0,0,0)])
    security_ids.index.names = ['assetid']
    
    ids_df = entity_ids.join(security_ids, how = 'outer').reset_index()
    ids_df = ids_df[[id_from, id_to]].drop_duplicates().set_index([id_from])

    data_df_converted =  data_df.join(ids_df, on = id_from, how = 'left')
    
    if keep_id_from:
        data_df_converted = data_df_converted.reset_index([id_from]) 
    else:
        data_df_converted = data_df_converted.reset_index([id_from], drop = True)
        
    if index_names[0] == 'assetid' and match_to_primary == True:
        data_df_converted = data_df_converted.rename(columns={'primary_assetid' : 'assetid'})
 
    if drop_unmatched:
        data_df_converted = data_df_converted[~data_df_converted[index_names[0]].isnull()]
        
        if (index_names[0] == 'assetid') or (index_names[0] == 'primary_assetid'):
            data_df_converted[index_names[0]] = data_df_converted[index_names[0]].astype(int)

    data_df_converted = data_df_converted.reset_index().set_index(index_names)
    
    if drop_duplicates == True:
        data_df_converted = data_df_converted[~data_df_converted.index.duplicated(keep='first')].sort_index()
    
    if 'index' in data_df_converted.columns:
        data_df_converted = data_df_converted.drop(columns = 'index')
    
    return data_df_converted


def assetid_to_entityid(data_df: pd.DataFrame, keep_ids:bool = False, drop_unmatched:bool = True):
    """
    Converts a dataframe indexed on assetid to entity_ids

    Args:
        data_df (Pandas DataFrame): input DataFrame.
        keep_ids (bool): keep assetid as a seperate column
        drop_unmatched (bool): drop rows that do not map to an entity_id

    Returns:
        Pandas Dataframe: Dataframe with entity_id

    """
    
    # load entity_ids
    entity_ids = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)]).reset_index().set_index('fsentityid')
    entity_ids.index.names = ['entity_id']
    entity_ids = entity_ids.reset_index().set_index(PrimaryKey.assetid)
    entity_ids.index.names = ['assetid']
    
    # convert assetid to entity_id
    data_df_converted = data_df.join(entity_ids['entity_id'], on = 'assetid')
    
    if drop_unmatched:
        data_df_converted = data_df_converted.dropna(subset = ['entity_id'])
    
    if keep_ids:
        data_df_converted = data_df_converted.reset_index()
    else:
        data_df_converted = data_df_converted.reset_index().drop(columns = 'assetid')
    
    return data_df_converted


def entityid_to_assetid(data_df: pd.DataFrame, keep_ids:bool = False, drop_unmatched:bool = True, match_to_primary:bool = True):
    """
    Converts a dataframe indexed on entity_ids to assetid

    Args:
        data_df (Pandas DataFrame): input DataFrame.
        keep_ids (bool): keep entity_id as a seperate column
        drop_unmatched (bool): drop rows that do not map to an assetid
        match_to_primary (bool): convert to primary assetid

    Returns:
        Pandas Dataframe: Dataframe with assetid

    """
    
    # convert entity_id to primaryassetid
    entity_ids = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)]).reset_index().set_index('fsentityid')
    entity_ids.index.names = ['entity_id']
    if match_to_primary:
        entity_ids = entity_ids.rename(columns = {'primary_assetid':'assetid'})
    else:
        entity_ids = entity_ids.rename(columns = {PrimaryKey.assetid:'assetid'})
        
    entity_ids = entity_ids['assetid'].drop_duplicates()
    
    # convert entity_id to assetid
    data_df_converted = data_df.join(entity_ids, on = 'entity_id')
    
    if drop_unmatched:
        data_df_converted = data_df_converted.dropna(subset = ['assetid'])
        data_df_converted['assetid'] = data_df_converted['assetid'].astype(int)
    else:
        data_df_converted['assetid'] = data_df_converted['assetid'].astype(float)
    
    if keep_ids:
        data_df_converted = data_df_converted.reset_index()
    else:
        data_df_converted = data_df_converted.reset_index().drop(columns = 'entity_id')
    
    
    return data_df_converted



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
    for col in np.unique(dict_clean.index):
        if len(dict_clean.loc[col, [col_name]]) == 1:
            dict_[col] = dict_clean.loc[col, [col_name]].values.tolist()
        elif len(dict_clean.loc[col, [col_name]]) > 1:
            dict_[col] = list(set(dict_clean.loc[col, col_name].values.tolist()))
    del dict_clean
    clean.collect()
    
    if not dict_:
        logger.info("WARNING: Dict is empty after cleaning.")
    else:
        count = len(dict_.keys())
        logger.info("Success! Dictionary of {0} features cleaned.".format(str(count)))

    return dict_


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
            df_clean[col] = pd.to_numeric(df_clean[col],downcast=dtype)
    clean.collect()

    # Drop empty columns
    if drop_cols:
        df_clean.dropna(how="all",axis=1,inplace=True)

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
        date_max = new_df.index.get_level_values(PrimaryKey.date).max()
        idx = pd.IndexSlice
        count = len(new_df.loc[idx[:, date_max], :].index.get_level_values(PrimaryKey.assetid).unique())
        
        logger.info("Success! DataFrame rounded: {0} assets as of {1}.".format(str(count), str(date_max.date())))

    return new_df



def make_assets_primary(data_df: pd.DataFrame):
    """
    Converts datafrale with assetid column to ensure all assetids are primary

    Args:
        data_df (pd.DataFrame): pd.DataFrame to convert

    Returns:
        pd.DataFrame: converted dataframe with primary assetids only
    """
    client = DBClient(
    db_engine=psycopg2, config=db_config
    )
    
    conn = client.get_connection()
    
    # load primary assetids
    assetinfo_sql = "SELECT asset_id as assetid, primary_asset_id FROM assetinfo"
    assetinfo = pd.read_sql(assetinfo_sql, conn)
    # assetinfo = ilf.get_sql_st(assetinfo_sql)

    data_df_primary =  pd.merge(data_df, assetinfo, on = 'assetid', how = 'left')

    # drop non-primary assetids
    data_df_primary = data_df_primary.loc[data_df_primary['assetid'] == data_df_primary['primary_asset_id']]
    data_df_primary = data_df_primary.loc[~data_df_primary['assetid'].isnull()].drop(columns = 'primary_asset_id')
    
    return data_df_primary

def remove_inactive(df, date_idx):
    """
    Removes inactive listings based on specified date column.
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
    client = DBClient(
    db_engine=psycopg2, config=db_config
    )

    active_sql = """
    SELECT a.asset_id AS assetid,
            CAST(a.first_non_zero_volume_date - INTERVAL '30 day' AS date) AS first_date,
            CAST(a.last_non_zero_volume_date + INTERVAL '30 day' AS date) AS last_date
    FROM asset_volume_start_end_dates a
    LEFT JOIN assetinfo b ON a.asset_id = b.asset_id
    WHERE b.primary_asset_id = a.asset_id
      AND a.asset_id IS NOT NULL
      AND b.name != 'Government'
    """

    conn = client.get_connection()
    active_raw_df = pd.read_sql(active_sql, conn, parse_dates=["first_date", "last_date"])
    
    active_df = clean_df(active_raw_df, "assetid")
    active_df.name = "active"
    
    df = df.join(active_df,how="left",sort=True, on = 'assetid')
    df = df.loc[(df.index.get_level_values(date_idx)>=df.first_date)&
                                    (df.index.get_level_values(date_idx)<=df.last_date)]
    df.drop(["first_date","last_date"],axis=1,inplace=True)

    return df

 
def clean_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises all null types to np.nan in a given pandas dataframe
 
    Args:
        df (pd.DataFrame): input dataframe to be cleaned
 
    Returns:
        pd.DataFrame: only containing np.nans for null values
    """
    df = df.replace({
            'NaN': np.nan,
            'nan': np.nan,
            'None': np.nan,
            None: np.nan,
            '': np.nan
            })
   
    return df

def cast_types_and_index(df: pd.DataFrame, index_dict: dict, make_numeric: bool = False) -> pd.DataFrame:
    '''
    Cast specified columns to the respective specified data types and sets them as multiindex.
    Remove duplicates.
    If called, it will also convert numeral strings back to numeric. ATTENTION: This will destroy date columns.
    
    Args:
        df (pd.DataFrame): input dataframe to be reindexed
        index_dict (dict): dictionary of columns to be indexed: {'col_name': 'col_data_type'}
        make_numeric (bool): statement, if numeral string should be converted to numerics

    Returns:
        pd.DataFrame: with new index
    '''
    df = df.drop_duplicates()

    for key, value in index_dict.items():
        df[key] = df[key].astype(value)

    if make_numeric:
        # attempt to convert numerics currently as strings back to numerics
        # note this will destroy date cols
        df = df.apply(pd.to_numeric, errors='ignore')

    df = df.set_index([key for key in index_dict])
    return df




def map_ids_to_assetid(df: pd.DataFrame, drop_unmatched = True):
    """
    Takes Dataframe containing fsentityid, cusip, sedol and isin cols, and adds
    assetid col that maps  the primary assetid for the listed identifiers

    Args:
        df (DataFrame): Dataframe containing fsentityid, cusip, sedol and isin cols

    Returns: Pandas DataFrame
    """
    df_mapped = df.copy()
    
    # convert columns to lower case
    df_mapped.columns = [col.lower() for col in df_mapped.columns]
    
    # combine ticker and exchange columns
    if 'ticker' in df_mapped.columns and 'exchange' in df_mapped.columns:
        df_mapped['ticker_exchange'] = df_mapped['ticker'].astype(str) + '-' + df_mapped['exchange'].astype(str)
    df_mapped = df_mapped.rename(columns = {'entity_id':'fsentityid'})
    
    # add assetid column to dataframe
    df_mapped['assetid'] = np.nan
    df_mapped = df_mapped.astype(object)
    
    # load entity ids and set primary_assetid to index
    entity_ids = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)])
    
    # load security ids and set primary_assetid to index
    security_ids = lt.get_meta_data(apps['assetinfo_security_id'][(1,0,0,0)])
    security_ids = security_ids.join(entity_ids, how = 'left')
    security_ids = security_ids.reset_index(drop = True) 
    security_ids = security_ids.rename(columns = {'primary_assetid':'assetid'})
    
    # fill in assetids based on identifiers
    for col in ['fsentityid', 'cusip', 'sedol', 'isin', 'ticker_exchange']:
        if col in df_mapped.columns:
            df_mapped.update(df_mapped.drop('assetid', 1).dropna(subset = [col]).merge(security_ids[['assetid', col]].drop_duplicates(),
                                                                     'left', col))
        
    if drop_unmatched:
        df_mapped=df_mapped.dropna(subset =['assetid'], how='all')
        df_mapped['assetid'] = df_mapped['assetid'].astype(int)
    else:
        df_mapped['assetid'] = df_mapped['assetid'].astype(float)
        
    df_mapped = df_mapped.drop(columns = ['ticker_exchange'])

    return df_mapped