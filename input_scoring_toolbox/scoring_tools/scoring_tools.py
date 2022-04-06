#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:56:57 2021

@author: roandufeu
"""


import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import scipy.stats as stats

import logging
logger = logging.getLogger(__name__)

import math
import scipy as sp
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

import sys



def cumulative_probability_single_date(metric_values: pd.DataFrame, polarity: int, 
                                       no_bins: int = 1000, trapz: bool = True):
    """
    Converts dataframe with single date and a single continuous metric into a 0-100 score,
    by binning data into bins then cumulatively summing and normalizing to generate a 0-100 
    cumulative probability for each bin.

    Args:
        metric_values (DataFrame): DataFrame containing single column with continuous data
        polarities: series mapping metrics to 'Positive' or 'Negative' polarity
        no_bins: number of bins to divide data into

    Returns:
        Pandas Series: Series containing scored continuous metrics

    """

    metric_values_nonan = polarity*metric_values[~np.isnan(metric_values)]    
    probs, bins = np.histogram(metric_values_nonan, bins = (no_bins ))
    
    
    if trapz == True:
        probs = np.append(0, probs)
    
        probs = probs/probs.sum()
        probs_cumulative = 100*cumtrapz(probs)
        
    else:
        probs = probs/probs.sum()
        probs_cumulative = 100*probs.cumsum()
    
    # add small amout to final bin to stop upper border values getting nan
    bins = np.nextafter(bins, bins + (bins == bins[-1]))
    value_bin_nos = np.digitize(polarity*metric_values,bins)
    
    probs_cumulative = np.append(probs_cumulative, np.nan)
    
    value_scores =  probs_cumulative[value_bin_nos -1]
    
    return pd.Series(index = metric_values.index, data = value_scores)


def cumulative_probability_scoring(data_df: pd.DataFrame, polarities: dict,  
                                    date_index:str = 'date', no_bins: int = 1000, trapz: bool = True):
    """
    Converts dataframe of continuous data to 0-100 scores, by binning data into n bins 
    then cumulatively summing and normalizing to generate a 0-100 cumulative probability
    for each bin

    Args:
        data_df (DataFrame): DataFrame where columns are continuous metrics
        polarities: dict mapping metrics to 'Positive' or 'Negative' polarity
        no_bins: number of bins to divide data into

    Returns:
        Pandas Series: Series containing scored continuous metrics

    """
    
    data_df = data_df.astype(float)
    
    data_df_scored = pd.DataFrame(index = data_df.index)
    
    for col in data_df.columns:   
        
        polarity = polarities[col]
        
        if str(polarity) == 'nan':
            data_df_scored[col]  = np.nan
        elif polarity ==  'Positive'  :   
            data_df_scored[col] = data_df[col].groupby(level = date_index).apply(lambda x: cumulative_probability_single_date(x, 1, no_bins, trapz)) 
        elif polarity ==  'Negative'   :  
            data_df_scored[col] = data_df[col].groupby(level = date_index).apply(lambda x: cumulative_probability_single_date(x, -1, no_bins, trapz)) 


    return data_df_scored


def modified_percentile_rank_single_date(data_df: pd.DataFrame, polarities:dict):
    """
   Converts dataframe of categorical (float) or binary 1/0 data to 0-100 scores
   using the modified percentile methodology.

    Args:
        data_df (DataFrame): DataFrame where columns are all categorical floats.
        polarities: series mapping metrics to 'Positive' or 'Negative' polarity

    Returns:
        Pandas Dataframe: Dataframe containing scored binary metrics

    """
    
    data_df_scored = pd.DataFrame(np.nan, index=data_df.index, columns=data_df.columns)
    
    # loop through metrics
    for metric_name in data_df.columns:
        # select metric name and data
        metric_data = data_df[metric_name]
        
        # score metric
        codes, uniques = pd.factorize(metric_data, sort=True)
        ncat = uniques.size
        
        cum_freq = 0
        for j in range(ncat) :
            if polarities[metric_name] == 'Positive' :
                temp_data = metric_data[metric_data==uniques[j]]
            elif polarities[metric_name] == 'Negative' :
                temp_data = metric_data[metric_data==uniques[ncat-1-j]]
            else :
                logger.info('Check for typos in the polaritity of ' + metric_name)
            cat_freq = temp_data.count() / metric_data.count()
            avg_cat_freq = (cum_freq + (cum_freq + cat_freq)) / 2
            cat_score = 100 * avg_cat_freq
            data_df_scored.loc[temp_data.index, metric_name] = cat_score
            cum_freq += cat_freq
        
        return data_df_scored
    
def modified_percentile_rank_scoring(data_df: pd.DataFrame, polarities: dict, date_index:str = 'date'):
    """
    Converts dataframe of categorical (float) or binary 1/0 data to 0-100 scores
    using the modified percentile methodology.

    Args:
        data_df (DataFrame): DataFrame where columns are all categorical floats.
        polarities: series mapping metrics to 'Positive' or 'Negative' polarity
        date_index: name of date index to group on

    Returns:
        Pandas Dataframe: Dataframe containing scored binary metrics

    """
    
    data_df_scored = pd.DataFrame(index = data_df.index)
    
    
    for col in data_df.columns:
        data_df_scored[col] = data_df[col].to_frame().groupby('date').apply(lambda x: modified_percentile_rank_single_date(x,polarities))

    return data_df_scored

# Scale/rank indicators
def percentile_rank_scoring_single_date(x, rank=True):
    """
    Ranks input data amongst all values for the current input column between 0 to 100
    and/or removes data with no information (all data values are the same)
    Args:
        x (Pandas Series): data column to rank
        rank (bool, default = True): If False, data is not ranked
    Returns:
        Pandas Series: Ranked data
    """

    if x.nunique() > 1 and rank:
        x_rank = x.rank(pct=True) * 100
    elif x.nunique() > 1 and not rank:
        x_rank = x.copy()
    else:
        x_rank = x.copy()
        x_rank[:] = np.nan

    return x_rank

def percentile_rank_scoring(data_df:pd.DataFrame, polarities: dict, date_index:str = 'date' ):
    """
    Ranks dataframe of float metrics to 0 to 100 scores, as is used on IR data in 262
    Args:
        data_df (DataFrame): DataFrame where columns are all float metrics to rank.
        ate_index: name of date index to group on
    Returns:
        Pandas Series: Ranked data
    """
    
    data_df_ranked = data_df.copy()
    
    for col in data_df:
        polarity = polarities[col]
        
        if str(polarity) == 'nan':
            data_df_ranked[col]  = np.nan
        elif polarity ==  'Positive'  :  
            data_df_ranked[col] = data_df[col].groupby(level=date_index).apply(lambda x: percentile_rank_scoring_single_date(x))
        elif polarity ==  'Negative'  :  
            data_df_ranked[col] = data_df[col].groupby(level=date_index).apply(lambda x: percentile_rank_scoring_single_date(-1*x))


    return data_df_ranked
    
def z_score_single_date(x):
    """
    Create z-score for a single date/ single set of datapoints

    Parameters
    ----------
    x : array of datapoints

    Returns
    -------
    array of datapoint z-scores

    """
    
    if x.std(ddof=0) == 0:
        return x * 0
    else:
        return (x - x.mean()) / x.std(ddof=0)

def z_score_cdf_scoring(data_df: pd.DataFrame, polarities: dict, date_index:str = 'date'):
    """
    Converts dataframe of floats to 0-100 scores, using z-score, then cdf,
    as is used to score TR data in 262

    Parameters
    ----------
    data_df (pd.DataFrame): Dataframe of float metrics (columns) to be scores
    date_index: name of date index to group on

    Returns
    -------
    Dataframe of scored columns.

    """
    
    data_df_scored = data_df.copy()

    for col in data_df.columns:
        polarity = polarities[col]
        if str(polarity) == 'nan':
            data_df_scored[col]  = np.nan
        elif polarity ==  'Positive'  :   
            data_df_scored[col] = data_df[col].groupby(level=date_index).apply(lambda x: z_score_single_date(x)) 
        elif polarity ==  'Negative'   :  
            data_df_scored[col] = data_df[col].groupby(level=date_index).apply(lambda x: z_score_single_date(-1*x)) 
        
    data_df_scored_ranked = pd.DataFrame(index=data_df_scored.index, columns=data_df_scored.columns,
                                  data=100 * stats.norm.cdf(data_df_scored))
    
    return data_df_scored_ranked

def kde_scoring(data_df: pd.DataFrame, polarities: dict, bounds: pd.DataFrame, date_index:str = 'date'):
    """
    Provides continuous metrics with a 0-100 score, based on the kde methodology

    Args:
        data_df (DataFrame): DataFrame where columns are continuous metrics
        polarities (dict): dict mapping metrics to 'Positive' or 'Negative' polarity
        bounds: DF of upper/lower bounds for each metric (column names 'Lower Bound' and 'Upper Bound')
        date_index: name of date_index

    Returns:
        Pandas DataFrame: DF containing scored continuous metrics

    """
    pol_cont = pd.Series(polarities)
    pol_cont.index.name = 'S-Ray Key'
    pol_cont.name = 'Polarity'
    
    pol_cont = pol_cont.sort_index()
    bounds = bounds.sort_index()
    data_df = data_df.sort_index(axis = 1)
    
    index_check = (pol_cont.index == data_df.columns).all()
    
    index_check2 = (bounds.index == data_df.columns).all()

    if not (index_check and index_check2):
         logger.exception('KDE scoring attempted with metadata that does not exactly match the metrics passed.')
         exit_code = 103
         print("Exiting with exit code {0}.".format(exit_code))
         sys.exit(exit_code)
    
    data_df_scored = data_df.copy()
    

    data_df_scored = data_df.groupby(level=date_index).apply(lambda x: kde_scoring_single_date(x, pol_cont, bounds)) 
    
    return data_df_scored


def kde_scoring_single_date(data_df: pd.DataFrame, pol_cont: pd.Series, bounds: pd.DataFrame):
    """
    Provides continuous metrics for a single date with a 0-100 score, based on the kde methodology

    Args:
        data_df (DataFrame): DataFrame where columns are continuous metrics
        polarities (Series): series of metric polarities, 'Positive' or 'Negative'
        bounds: DF of upper/lower bounds for each metric (column names 'Lower Bound' and 'Upper Bound')
        date_index: name of date_index

    Returns:
        Pandas Series: Series containing scored continuous metrics for single date

    """
    
    nfp = 1001  # number of fixed points for the EWMA CDF calcs

    
    mkeys_cont = pd.Series(data=data_df.columns, name = 'S-Ray Key')
    lb_cont = bounds['Lower Bound']
    ub_cont = bounds['Upper Bound']

    corrected_data, kurt, mmax, mmin, logflag, startpoint, endpoint, fp_x = calc_kde_params(data_df, mkeys_cont, 
    lb_cont, ub_cont, pol_cont)


    # now figure out which metrics/columns to push to binary v continuous
    cont_data = corrected_data[mkeys_cont]
    
    
    #####################
    ### Input scoring ###
    #####################
    
    
    # continuous scoring
    cdata_in = cont_data.copy()
    lflag = logflag.copy()
    lbd = lb_cont.copy()
    ubd = ub_cont.copy()
    pol = pol_cont.copy()

    # number of continuous metrics
    ncontmets = len(cdata_in.columns)
    
    # note nfp is number of fixed points for the KDE calcs 
    # as python KDEs require a set of fixed points to compute
    cdf_fp = pd.DataFrame(np.zeros((nfp, ncontmets)),columns=cdata_in.columns)
    
    mscores = pd.DataFrame(np.nan, index=cdata_in.index, columns=cdata_in.columns)
    
    for k in range(ncontmets):
        met_tag = cdata_in.columns[k]
        mdata = cdata_in[met_tag].astype(float)
        num_data = mdata[~np.isnan(mdata)]
    
        # if all data is NULL nothing to score 
        if mdata.isna().all():
            continue
        
        # or if all data is the same assign value of 50 
        if len(num_data.value_counts()) == 1:
            mscores.loc[num_data.index, met_tag]  = 50.0
            continue
    
        # first separate out lower and upper bound responses (deal with possible inflation)
        if not np.isnan(lbd[met_tag]) :
            # identify lower bound data
            lb_data = num_data[num_data==lbd[met_tag]]
            # remove lower bound data from num data
            num_data = num_data[num_data!=lbd[met_tag]]
        else :
            # if no lower bound for metric persist nan    
            lb_data = np.nan
        if not np.isnan(ubd[met_tag]) :
            ub_data = num_data[num_data==ubd[met_tag]]
            num_data = num_data[num_data!=ubd[met_tag]]
        else :
            ub_data = np.nan
        
        # Calculate boundary scores FIRST 
        lb_count = 0
        ub_count = 0
        if not np.isnan(lb_data).all() :
            lb_count += lb_data.size
        if not np.isnan(ub_data).all() :
            ub_count += ub_data.size
        lb_freq = lb_count / (lb_count + ub_count + num_data.size)
        ub_freq = ub_count / (lb_count + ub_count + num_data.size)
        bd_freq = lb_freq + ub_freq
        
        # check for polarity type and calc scores accordingly
        if pol[met_tag] == 'Positive' :
            mscore_lb = 100 * (lb_freq / 2)
            mscore_ub = 100 * (1 - (ub_freq / 2))
        elif pol[met_tag] == 'Negative' :
            mscore_lb = 100 * (1 - (lb_freq / 2))
            mscore_ub = 100 * (ub_freq / 2)
        elif pol[met_tag] == 'Midpoint' :
            mscore_lb = 100 * (bd_freq / 2)
            mscore_ub = 100 * (bd_freq / 2)
        else :
            print('Check for typos in the polarity of ' + met_tag)
    
    
        # map lb and ub scores back to the relevant assetids, then map everything
        # back to the incoming dataframe's assetids and metric keys
        if not np.isnan(lb_data).all() :
            # convert mscore_lb from single value back into series across entity_ids it applies to
            mscore_lb_data = lb_data.replace(lbd[met_tag], mscore_lb)
            mscores.loc[mscore_lb_data.index, met_tag] = mscore_lb_data
        if not np.isnan(ub_data).all() :
            mscore_ub_data = ub_data.replace(ubd[met_tag], mscore_ub)
            mscores.loc[mscore_ub_data.index, met_tag] = mscore_ub_data
    
    
        #  if no numeric data not at the bounds left, skip to recombination 
        # if there is no data left, skip to next metric
        if len(num_data) == 0:    
            continue
        
    
        #  if all remaining non-boundary data is the same assign value of 0.5 
        # DONE
        elif len(num_data.value_counts()) == 1:
            
            cdf_actdata = pd.Series( data = 0.5, index = num_data.index, 
                                         name = num_data.name,
                                         dtype = num_data.dtype)
        
        else:
        
            # now take logs on fat-tailed, non-negative, unbounded metrics
            # NOTE when logs are taken -ve values are generated
            if lflag[k] :
                num_data = np.log(num_data)
        
            # recast the midpoint polarity data
            if pol[met_tag] == 'Midpoint' :
                mp = (ubd[met_tag] + lbd[met_tag]) / 2
                num_data = mp - abs(mp - num_data)
            
            # calculate Silverman's optimal bandwidth
            sbd = calc_silv_band(num_data)
            
            # need to sort data into ascending order for numerical integration later
            num_data = num_data.sort_values()
            
            # convert data to correctly sized numpy array for KD fcn
            nd_array = np.reshape(num_data.to_numpy(copy=True),(-1,1))
            
            # if sbd = 0 use std instead. If std is 0 or null use 1 -note 
            #  this should never happen as such cases (no values/ one value/ all 
            # values the same) are dealt with already above
            num_data_std = num_data.std()
            if sbd == 0.0 and pd.isna(num_data_std):
                sbd = 1
            elif sbd == 0.0 and num_data_std == 0.0:
                sbd = 1
            elif sbd == 0.0:
                sbd = num_data_std
                
        
            # use kde to find empirical pdf and numerically integrate the cdf
            kde = KernelDensity(kernel='gaussian', bandwidth=sbd).fit(nd_array)
            kde_lnpdf = kde.score_samples(nd_array)
            # KD generates a log pdf(?) so take exp to generate a proper pdf
            kde_pdf = np.exp(kde_lnpdf)
            # take integral to go from pdf -> cdf
            kde_cdf = sp.integrate.cumulative_trapezoid(kde_pdf, num_data, initial=0)
            # 'trim' cdf so that no values are less than zero nor greater than one
            kde_cdf[kde_cdf<0] = 0
            kde_cdf[kde_cdf>1] = 1
            # now need to interpolate a cdf using fixed points
            cdf_fp[met_tag] = lin_interp(num_data.copy(), fp_x[met_tag], kde_cdf.copy())
            cdf_fp[met_tag][cdf_fp[met_tag]<0] = 0
            cdf_fp[met_tag][cdf_fp[met_tag]>1] = 1
            
        
            # map fixed pt CDF back to actual data 1st
            cdf_actdata = lin_interp(fp_x[met_tag], num_data.copy(), cdf_fp[met_tag])
            
            
        # check for polarity type and calc scores accordingly
        if pol[met_tag] == 'Positive' :
            mscore_num = 100 * (((1 - bd_freq) * cdf_actdata) +  lb_freq)
        elif pol[met_tag] == 'Negative' :
            mscore_num = 100 * (((1 - bd_freq) * (1 - cdf_actdata)) +  ub_freq)
        elif pol[met_tag] == 'Midpoint' :
            mscore_num = 100 * (((1-bd_freq) * cdf_actdata) + bd_freq)
        else :
            print('Check for typos in the polarity of ' + met_tag)
        
        # map all non upper / lower bound data back
        mscores.loc[mscore_num.index, met_tag] = mscore_num
    
    return mscores

   
def calc_kde_params(data_df: pd.DataFrame, mkeys: pd.Series, lb: pd.Series, ub: pd.Series, pol: pd.Series):
    kurt_threshold = 2 
    nfp = 1001 
    # initialise variables
    kurt = pd.Series(np.zeros(len(mkeys)))
    mmax = kurt.copy()
    mmin = kurt.copy()
    logflag = np.zeros(len(mkeys), dtype=bool)
    startpoint = kurt.copy()
    endpoint = kurt.copy()
    corrected_data = data_df.copy()

    # loop through individual metrics to reduce overall run time
    for i in tqdm(range(len(mkeys))) :
        mkey = mkeys.iloc[i]
        met_data = data_df[mkey].astype(float)

        # reset data points outside the feasible range to the feasible boundary
        if not np.isnan(lb.iloc[i]) :
            met_data[met_data<lb.iloc[i]] = lb.iloc[i]
        if not np.isnan(ub.iloc[i]) :
            met_data[met_data>ub.iloc[i]] = ub.iloc[i]
        
        # save out corrected data
        corrected_data[mkey] = met_data

        kurt[i] = sp.stats.kurtosis(met_data, nan_policy='omit')
        # if data is leptokurtic (fat-tailed), non-negative and unbounded from above, take logs
        if kurt[i] > kurt_threshold and lb.iloc[i] >= 0 and np.isnan(ub.iloc[i]) :
            mmax[i] = np.log(met_data[met_data>0]).max()
            mmin[i] = np.log(met_data[met_data>0]).min()
            logflag[i] = True
        elif pol.iloc[i] == 'Midpoint' :
            mp = (lb.iloc[i] + ub.iloc[i]) / 2
            met_data_tform = mp - abs(mp - met_data)
            mmax[i] = met_data_tform.max()
            mmin[i] = met_data_tform.min()
            logflag[i] = False
        else :
            mmax[i] = met_data.max()
            mmin[i] = met_data.min()
            logflag[i] = False

        if not np.isnan(mmin[i]):
            startpoint[i] = math.floor(mmin[i])
        else:
            startpoint[i] = np.nan

        if not np.isnan(mmax[i]):
            endpoint[i] = math.ceil(mmax[i])
        else:
            endpoint[i] = np.nan
        
        

    fp_x = pd.DataFrame(np.linspace(startpoint, endpoint, num=nfp),columns=mkeys)

    return corrected_data, kurt, mmax, mmin, logflag, startpoint, endpoint, fp_x


def calc_silv_band(numdata: pd.Series) -> int:
    iqr = sp.stats.iqr(numdata)
    std = np.std(numdata, ddof=1)
    sbd = 0.9 * min(iqr, std) * (numdata.size ** (-1/5))
    return sbd

def lin_interp(x: pd.Series, x_new: pd.Series, y: pd.Series) -> pd.Series: 
    # uses linear interpolation to approx. f in y=f(x) 
    # and map to y'= f(x'). Assumes x and x_new are already sorted in ascending order
    # first bind x and y together in a dataframe and remove duplicates in x
    datain = pd.DataFrame({'x':x,'y':y})
    datain.drop_duplicates(subset='x', inplace=True)
    x_new = pd.Series(x_new)
    x_in = pd.Series(datain['x'])
    y_in = pd.Series(datain['y'])
    n_in = len(datain)
    n_out = len(x_new)
    y_new = pd.Series(np.zeros(n_out))
    grad = pd.Series(np.zeros(n_in-1))

    # calculate the gradient for x and y (will have n-1 elements)
    for i in range(n_in - 1) :
        grad[i] = (y_in.iloc[i+1]-y_in.iloc[i]) / (x_in.iloc[i+1]-x_in.iloc[i])
        
    for i in range(n_out) :
        if x_new.iloc[i] < x_in.iloc[0] :
            y_new[i] = grad[0]*(x_new.iloc[i]-x_in.iloc[0])+y_in.iloc[0]
        elif x_new.iloc[i] >= x_in.iloc[n_in-1] :
            y_new[i] = grad[n_in-2]*(x_new.iloc[i]-x_in.iloc[n_in-1])+y_in.iloc[n_in-1]
        else :
            tempdiff = (x_in - x_new.iloc[i]).reset_index(drop=True)
            lwr_val = max(tempdiff[tempdiff<=0])
            grad_loc = tempdiff[tempdiff==lwr_val].index.to_numpy()
            grad_loc = grad_loc[0]
            y_new[i] = grad[grad_loc]*(x_new.iloc[i]-x_in.iloc[grad_loc])+y_in.iloc[grad_loc]
    # assign index and column names in x_new to y_new
    y_new.index = x_new.index
    y_new.name = x_new.name
    return y_new


