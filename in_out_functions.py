import logging
import os
import typing

import numpy as np
import pandas as pd
from sray_db.apps import apps
from sray_db.apps.pk import PrimaryKey
from sray_db.broker import DataBroker
from sray_db.query import Get
from sray_db.query import Put

logger = logging.getLogger(__name__)


def load_app_data(app_name: str, app_version: tuple, start_date: str = None, end_date: str = None,
                  assetids: typing.Union[None, typing.List[int]] = None,
                  as_of: typing.Union[None, pd.Timestamp] = None) -> pd.DataFrame:
    
    fields = list(apps[app_name][app_version].values())
    
    db = DataBroker()
    
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp('2013-01-01')
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.utcnow().replace(tzinfo=None)
    
    dates = pd.date_range(start, end, freq='W', name=PrimaryKey.date)
    load_idx = [dates]
    
    if assetids:
        assets = pd.Index(assetids, name=PrimaryKey.assetid)
        load_idx.append(assets)
        
    get_query = Get(fields, load_idx=load_idx, as_of=as_of)
    data = db.query(get_query)
    
    field_names = apps[app_name][app_version].keys()
    
    data.columns = field_names
    
    logger.info(f"Loaded dataset of shape {data.shape} from {app_name}")
    
    if data.empty:
        logger.warning(f"No data loaded for {app_name}")
        return

    data = data.reset_index()

    return data


def get_csv_from_file(fname: str, date_col: typing.Union[None, typing.List[str]] = None) -> pd.DataFrame:
    logger.debug("Retrieving file from path {0}".format(fname))

    full_path = os.path.join(os.path.dirname(__file__), fname)
    file_out = pd.read_csv(full_path, parse_dates=date_col, dayfirst=True)

    return file_out
