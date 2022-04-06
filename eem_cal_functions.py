#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:11:31 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_dev")

import pandas as pd
from sklearn import metrics

import xgboost as xgb
import optuna

from sray_db.apps.pk import PrimaryKey

import _config_ as config

def xgBoostyBoost (X_train, y_train, X_test, y_test, X_real):
    
    y_train = y_train.loc[y_train['industry'] == X_train['industry'].iloc[0]]
    X_test = X_test.loc[X_test['industry'] == X_train['industry'].iloc[0]]
    y_test = y_test.loc[y_test['industry'] == X_train['industry'].iloc[0]]
    
    X_real = X_real.loc[X_real['industry'] == X_train['industry'].iloc[0]]
    
    if len(X_train) > config.min_datapoints and len(X_test) > 0:
        
        study = optuna.create_study(direction="minimize")
    
        def objective(trial):
            """
            Objective function to tune an `XGBRegressor` model.
            """

            params = {
                'n_estimators': trial.suggest_int("n_estimators", 1, 10000),
                'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-8, 100.0),
                'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
                'max_depth': trial.suggest_int("max_depth", 2, 9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                }


            model = xgb.XGBRegressor(
                booster="dart",
                objective="reg:squarederror",
                random_state=42,
                **params
                )

            return train_model_for_study(X_train, y_train, X_test, y_test, model)

        study.optimize(objective, n_trials=config.n_trials, timeout=600)
    
        bestboostyboost = study.best_params

        '''
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                                  max_depth = 5, alpha = 10, n_estimators = 10)
        '''
    
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', **bestboostyboost)
        
        xg_reg.fit(X_train[config.variables], y_train.em_true)
        
        if config.test == True:
            result = X_test.groupby([PrimaryKey.assetid]).apply(lambda x: xg_reg.predict(x[config.variables]))
        else:
            result = X_real.groupby([PrimaryKey.assetid]).apply(lambda x: xg_reg.predict(x[config.variables]))
        
        result = abs(pd.DataFrame(result))  
        result['datapoints'] = len(X_train)
        
        return result
    
def train_model_for_study(X_train, y_train, X_test, y_test, model):
    
    model.fit(
        X_train[config.variables], 
        y_train.em_true,
    )

    yhat = model.predict(X_test[config.variables])
    
    return metrics.mean_squared_error(y_test.em_true, yhat, squared=False)