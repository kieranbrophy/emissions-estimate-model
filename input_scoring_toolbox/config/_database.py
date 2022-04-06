"""
Instantiate DB creds from .env file
"""
import logging
import os
import sys
from typing import List

def get_missing_env_vars(env_var_list: List[str]) -> List[str]:
    missing_vars = []
    for var_name in env_var_list:
        env_var = os.getenv(var_name)
        if not env_var:
            missing_vars.append(var_name)
    return missing_vars

__all__ = [
    "PG_HOST",
    "PG_PORT",
    "PG_USERNAME",
    "PG_PASSWORD",
    "PG_DB",
    "PG_USE_SSL",
    "PG_SSLROOTCERT",
    "PG_SSLKEY",
    "PG_SSLCERT",
    "db_config"
]

PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USERNAME = os.getenv("PG_USERNAME")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")

PG_USE_SSL=os.getenv("PG_USE_SSL")
PG_SSLROOTCERT=os.getenv("PG_SSLROOTCERT")
PG_SSLKEY=os.getenv("PG_SSLKEY")
PG_SSLCERT=os.getenv("PG_SSLCERT")

env_vars_to_validate = ["PG_HOST", 
                        "PG_PORT", 
                        "PG_USERNAME", 
                        "PG_PASSWORD",
                        "PG_DB", 
                        "PG_USE_SSL", 
                        "PG_SSLROOTCERT", 
                        "PG_SSLKEY", 
                        "PG_SSLCERT"]

missing_vars = get_missing_env_vars(env_vars_to_validate)
    
db_config = dict(
    dbname=PG_DB, 
    user=PG_USERNAME, 
    password=PG_PASSWORD, 
    host=PG_HOST, 
    port=PG_PORT,
    sslmode='require',
    sslrootcert = PG_SSLROOTCERT,
    sslcert = PG_SSLCERT,
    sslkey = PG_SSLKEY

)

if missing_vars:
    logging.exception(f"Missing environment variables {missing_vars}")
    sys.exit(1)