#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:22:04 2022

@author: kieran.brophyarabesque.com
"""
from sklearn import metrics


import _config_ as config
    
def train_model_for_study(X_train, y_train, X_test, y_test, model):
    
    model.fit(
        X_train[config.variables], 
        y_train.em_true,
    )

    yhat = model.predict(X_test[config.variables])
    
    return metrics.mean_squared_error(y_test.em_true, yhat, squared=False)