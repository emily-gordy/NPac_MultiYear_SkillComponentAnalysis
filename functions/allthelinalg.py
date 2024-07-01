#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:49:42 2024

@author: egordon4
"""

import xarray as xr
import glob
import numpy as np

from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.linalg import eig
from scipy.stats import skew, skewtest, percentileofscore

# def calculate_SC(y_pred_val,outputval,landmask):
    
#     predrectangle = y_pred_val[:,landmask]
#     verrectangle = outputval[:,landmask]
#     errrectangle = predrectangle-verrectangle
    
#     Se = np.cov(np.transpose(errrectangle))
#     Sv = np.cov(np.transpose(verrectangle))
    
#     eigvals,evecs = eig(Se,Sv)
#     bestinds = np.argsort(np.real(eigvals))

#     ivec = 0
    
#     evecsel = evecs[:,bestinds[ivec]]
    
#     if np.nansum(evecsel)<0:
#         evecsel = -1*evecsel
#     truecomponent = (1/len(evecsel))*np.matmul(evecsel,np.transpose(verrectangle))
#     #standardize component
#     truecomponent = (truecomponent-np.mean(truecomponent))/np.std(truecomponent)
    
#     bestpattern = (1/len(truecomponent))*np.matmul(np.transpose(verrectangle),truecomponent)
#     bestpattern_out = np.empty((outputval.shape[1],outputval.shape[2]))+np.nan
#     bestpattern_out[landmask]=bestpattern
    
#     return bestpattern_out

def index_timeseries(data,pattern,landmask):
    
    data_rect = data[:,landmask]
    patternvec = pattern[landmask]
    index = np.matmul(data_rect,patternvec)
    
    index = (index-np.mean(index))/np.std(index)
    
    return index

def corr_indextimeseries(bestpattern,truedata,preddata,outputval,y_pred_val,landmask):

    # SCstd_true,SCstd_pred = standardizations(outputval,y_pred_val,bestpattern,landmask)
    
    SC_index_true = index_timeseries(truedata,bestpattern,landmask)#/SCstd_true
    SC_index_pred = index_timeseries(preddata,bestpattern,landmask)#/SCstd_pred
    
    r,p = pearsonr(SC_index_true,SC_index_pred)
    
    return r,p

def err_indextimeseries(bestpattern,truedata,preddata,outputval,y_pred_val,landmask):

    SCstd_true,SCstd_pred = standardizations(outputval,y_pred_val,bestpattern,landmask)
    
    SC_index_true = index_timeseries(truedata,bestpattern,landmask)/SCstd_true
    SC_index_pred = index_timeseries(preddata,bestpattern,landmask)/SCstd_pred
    
    err = np.mean((SC_index_true-SC_index_pred)**2)
    
    return err

def calculate_SC_weighted(y_pred_val,outputval,landmask,weights):

    y_pred_val = y_pred_val*weights[np.newaxis,:,:]
    outputval = outputval*weights[np.newaxis,:,:]
    
    predrectangle = y_pred_val[:,landmask]
    verrectangle = outputval[:,landmask]
    errrectangle = predrectangle-verrectangle
    
    Se = np.cov(np.transpose(errrectangle))
    Sv = np.cov(np.transpose(verrectangle))
    
    eigvals,evecs = eig(Se,Sv)
    bestinds = np.argsort(np.real(eigvals))

    ivec = 0
    
    evecsel = evecs[:,bestinds[ivec]]
    
    if np.nansum(evecsel)<0:
        evecsel = -1*evecsel
    truecomponent = (1/len(evecsel))*np.matmul(evecsel,np.transpose(verrectangle))
    #standardize component
    truecomponent = (truecomponent-np.mean(truecomponent))/np.std(truecomponent)
    
    bestpattern = (1/len(truecomponent))*np.matmul(np.transpose(verrectangle),truecomponent)
    bestpattern_out = np.empty((outputval.shape[1],outputval.shape[2]))+np.nan
    bestpattern_out[landmask]=bestpattern
    
    return bestpattern_out

