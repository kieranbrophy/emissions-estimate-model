#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:11:31 2022

@author: kieran.brophyarabesque.com
"""

from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_prod")

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import median_absolute_error

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from joblib import dump
from joblib import load

from skl2onnx import convert_sklearn, update_registered_converter
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
                convert_xgboost)
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes

from skl2onnx.common.data_types import FloatTensorType

from sklearn.metrics import r2_score

import shap
import _config_ as config

from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt

'''
XG Boost calculation function
'''
def xgBoostyBoost (X_train, y_train, X_test, y_test, X_val, y_val, X_real, scope, run, sec_code, reg_code, check_sum):
        
    y_train = y_train['em_true']
    y_test = y_test['em_true']
    y_val = y_val['em_true']
    
    X_train = X_train.drop(columns=['industry','eco_sector','factset_sector','region','iso2','economic_sector','bespoke_economic_sector','bespoke_industry','fact_code','eco_code','ind_code','reg_code','iso_code'])
    X_test = X_test.drop(columns=['industry','eco_sector','factset_sector','region','iso2','economic_sector','bespoke_economic_sector','bespoke_industry','fact_code','eco_code','ind_code','reg_code','iso_code'])
    X_val = X_val.drop(columns=['industry','eco_sector','factset_sector','region','iso2','economic_sector','bespoke_economic_sector','bespoke_industry','fact_code','eco_code','ind_code','reg_code','iso_code'])
    X_real = X_real.drop(columns=['industry','eco_sector','factset_sector','region','iso2','economic_sector','bespoke_economic_sector','bespoke_industry','fact_code','eco_code','ind_code','reg_code','iso_code'])

    if len(y_val) >= config.data_thresh and len(y_test) > 0:
    
        def objective(trial):
            """
            Objective function to tune an `XGBRegressor` model.
            """
            params = {
                'max_depth' : trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators' : trial.suggest_int('n_estimators', 100, 500),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),           
                'subsample' : trial.suggest_float('subsample', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1.0, 5.0),
                }
            
            model = xgb.XGBRegressor(
                booster="gbtree",
                objective="reg:squarederror",
                random_state=0,
                **params,
                )
        
            model.fit(X_train, y_train)

            yhat = model.predict(X_test)
            
            return metrics.mean_squared_error(y_test, yhat, squared=False)
    
        '''
        Run XG Boost for best hyperparameters?
        '''
        if config.hyper_tune == True:

            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed = 42))
            study.optimize(objective, n_trials=config.n_trials, timeout=600)
            
            if config.show_opt_trial == True:
                
                plot_optimization_history(study)
                    
                plt.savefig('trials/run_' + str(run) + '/scope_' + str(scope) +'/pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.pdf', bbox_inches='tight')
            
            bestboostyboost = study.best_params
            
            xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', booster="gbtree",
                                      random_state=0, **bestboostyboost)

            update_registered_converter(
                xgb.XGBRegressor, 'XGBoostXGBRegressor',
                calculate_linear_regressor_output_shapes, convert_xgboost)

            pipe = Pipeline([('scaler', StandardScaler()), ('xgb', xg_reg)])
            
            pipe.fit(X_train, y_train)
            
            model_onnx = convert_sklearn(pipe, 'pipeline_xgboost',[('input', FloatTensorType([None, len(X_train.columns)]))],
                target_opset={'': 12, 'ai.onnx.ml': 2})
                     
            with open('xgb_saved_model/run_' + str(run) + '/scope_' + str(scope) +'/xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.onnx', "wb") as f:
                f.write(model_onnx.SerializeToString())
            
            # save model to file
            dump(xg_reg, 'xgb_saved_model/run_' + str(run) + '/scope_' + str(scope) +'/pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.dat')
            print('Saved model to: pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.dat')

        else:
            print('' + str(sec_code) + '_.dat')
            xg_reg = load('xgb_saved_model/run_20220701/scope_' + str(scope) +'/pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.dat')
            print('Loaded model from: pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.dat')

        xg_reg.fit(X_train, y_train)

        '''
        Validate estiamtes to get percentage error per sector
        '''
        val_pred = xg_reg.predict(X_val)
        val_result = pd.DataFrame(y_val)
        val_result['XGB_em_est'] = abs(val_pred)
                    
        MDAE = median_absolute_error(val_result['em_true'], val_result['XGB_em_est'])
        MDAPE = np.median((np.abs(np.subtract(val_result['em_true'], val_result['XGB_em_est'])/val_result['em_true']))) * 100
        r_sq = r2_score(val_result['em_true'], val_result['XGB_em_est'])
    
        '''
        Estimate emissions - including likely percentage error
        '''
        real_pred = xg_reg.predict(X_real)
        real_result = pd.DataFrame(X_real)
        real_result['XGB_em_est'] = abs(real_pred)
        
        real_result.loc[real_result['XGB_em_est'] < 0] = 0
                    
        real_result['XGB_error_est'] = MDAE
        real_result['XGB_per_error_est'] = MDAPE
        real_result['XGB_R_squared'] = r_sq
        real_result['XGB_datapoints_val'] = len(y_val)
        real_result['XGB_model_file'] = 'xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.onnx'
        
        real_result['em_test'] = y_test
        real_result['em_val'] = y_val
        
        if config.show_shap == True:
            explainer = shap.TreeExplainer(xg_reg)
            shap_values = explainer.shap_values(X_val, check_additivity=False)
            shap.summary_plot(shap_values, X_val, show=False)
            plt.savefig('shap/run_' + str(run) + '/scope_' + str(scope) +'/pima.xg_reg_' + str(sec_code) + '_' + str(reg_code) + '_' + str(check_sum) + '.pdf', bbox_inches='tight')

        result = real_result
                
        return result
