#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:49:42 2024

@author: egordon4
"""

import numpy as np

from scipy.stats import pearsonr,linregress
from scipy.linalg import eig

def index_timeseries(data,pattern,landmask):
    
    data_rect = data[:,landmask]
    patternvec = pattern[landmask]
    index = np.matmul(data_rect,patternvec)
    
    index = (index-np.mean(index))/np.std(index)
    
    return index

def corr_indextimeseries(bestpattern,truedata,preddata,landmask):

    SC_index_true = index_timeseries(truedata,bestpattern,landmask)
    SC_index_pred = index_timeseries(preddata,bestpattern,landmask)
    
    r,p = pearsonr(SC_index_true,SC_index_pred)
    
    return r,p

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

def mse(xpred,xtrue):

    return np.mean((xpred-xtrue)**2)

def some_simple_models(y_pred_test,outputtest,trainingval,bestpattern,landmask,lat,lon,run):

    a_SST = np.empty((len(lat),len(lon)))
    b_SST = np.empty((len(lat),len(lon)))

    SCtimeseries_training = index_timeseries(trainingval,bestpattern,landmask)
    SCtimeseries_pred = index_timeseries(y_pred_test,bestpattern,landmask)

    outputobs_pred = np.empty((len(SCtimeseries_pred),len(lat),len(lon)))
    outputobs_ACC = np.empty((len(lat),len(lon)))
    outputobs_p = np.empty((len(lat),len(lon)))

    outputobs_ACC_raw = np.empty((len(lat),len(lon)))
    outputobs_p_raw = np.empty((len(lat),len(lon)))
    persistence = np.empty((len(lat),len(lon)))

    MSSS_raw = np.empty((len(lat),len(lon)))
    MSSS_SC = np.empty((len(lat),len(lon)))

    for ilat,_ in enumerate(lat):
        for ilon,_ in enumerate(lon):

            a_SST[ilat,ilon],b_SST[ilat,ilon],_,_,_ = linregress(SCtimeseries_training,trainingval[:,ilat,ilon])

            outputobs_pred[:,ilat,ilon] = a_SST[ilat,ilon]*SCtimeseries_pred + b_SST[ilat,ilon]
            outputobs_ACC[ilat,ilon],outputobs_p[ilat,ilon] = pearsonr(outputobs_pred[:,ilat,ilon],outputtest[:,ilat,ilon])
            outputobs_ACC_raw[ilat,ilon],outputobs_p_raw[ilat,ilon] = pearsonr(outputtest[:,ilat,ilon],y_pred_test[:,ilat,ilon])
            persistence[ilat,ilon],_ = pearsonr(outputtest[:-run,ilat,ilon],outputtest[run:,ilat,ilon])

            MSSS_raw[ilat,ilon] = 1-(mse(outputtest[:,ilat,ilon],y_pred_test[:,ilat,ilon])/
                                     mse(outputtest[:-run,ilat,ilon],outputtest[run:,ilat,ilon]))
            MSSS_SC[ilat,ilon] = 1-(mse(outputobs_pred[:,ilat,ilon],outputtest[:,ilat,ilon])/
                                    mse(outputtest[:-run,ilat,ilon],outputtest[run:,ilat,ilon]))

    return outputobs_pred,outputobs_ACC,outputobs_ACC_raw,outputobs_p,outputobs_p_raw,persistence,MSSS_raw,MSSS_SC

def some_simple_models_tas(y_pred_test,outputTAS,trainingSST,trainingTAS,bestpattern,landmask,lat,lon,run):

    a_TAS = np.empty((len(lat),len(lon)))
    b_TAS = np.empty((len(lat),len(lon)))

    SCtimeseries_training = index_timeseries(trainingSST,bestpattern,landmask)
    SCtimeseries_pred = index_timeseries(y_pred_test,bestpattern,landmask)

    outputobs_pred = np.empty((len(SCtimeseries_pred),len(lat),len(lon)))
    outputobs_ACC = np.empty((len(lat),len(lon)))
    outputobs_p = np.empty((len(lat),len(lon)))

    MSSS_SC = np.empty((len(lat),len(lon)))

    persistence = np.empty((len(lat),len(lon)))

    for ilat,_ in enumerate(lat):
        for ilon,_ in enumerate(lon):

            a_TAS[ilat,ilon],b_TAS[ilat,ilon],_,_,_ = linregress(SCtimeseries_training,trainingTAS[:,ilat,ilon])

            outputobs_pred[:,ilat,ilon] = a_TAS[ilat,ilon]*SCtimeseries_pred + b_TAS[ilat,ilon]
            outputobs_ACC[ilat,ilon],outputobs_p[ilat,ilon] = pearsonr(outputobs_pred[:,ilat,ilon],outputTAS[:,ilat,ilon])
            # outputobs_ACC_raw[ilat,ilon],_ = pearsonr(outputTAS[:,ilat,ilon],y_pred_test[:,ilat,ilon])
            persistence[ilat,ilon],_ = pearsonr(outputTAS[:-run,ilat,ilon],outputTAS[run:,ilat,ilon])

            MSSS_SC[ilat,ilon] = 1-(mse(outputobs_pred[:,ilat,ilon],outputTAS[:,ilat,ilon])/
                                    mse(outputTAS[:-run,ilat,ilon],outputTAS[run:,ilat,ilon]))

    return outputobs_pred,outputobs_ACC,outputobs_p,MSSS_SC,persistence,a_TAS