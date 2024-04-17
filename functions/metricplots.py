#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:46:22 2024

@author: egordon4
"""

import xarray as xr
import glob
import numpy as np

from scipy.stats import norm
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import cmasher as cmr

def historyplot(history):
    plt.figure(figsize=(8, 3))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"],
             color='xkcd:dark turquoise', label='training')
    plt.plot(history.history["val_loss"],
             color='xkcd:burnt orange', label='validation')
    plt.ylim(0, 3)
    plt.title('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mean_absolute_error"],
             color='xkcd:dark turquoise')
    plt.plot(history.history["val_mean_absolute_error"],
             color='xkcd:burnt orange')
    plt.title('MAE')
    plt.ylim(0, 0.8)

    plt.tight_layout()
    plt.show()

def mapmetrics(y_pred,y_true,nvars,lon,lat,centre,title,settings):
     
    leadtime = settings["leadtime"]
    run = settings["run"]
    
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
     
    nlat = y_true.shape[1]
    nlon = y_true.shape[2]
    
    nyears = int(len(y_true)/nvars)
    y_auto = np.reshape(y_true,(nvars,nyears,nlat,nlon))
    
    y_out = np.copy(y_true)
     
    mseplot = np.mean((y_pred-y_out)**2, axis=0)
    outdims = y_pred.shape[1:]
    
    pearsonrplot = np.empty(outdims)+np.nan
    pearsonpplot = np.empty(outdims)+np.nan
    
    stdpred = np.std(y_pred, axis=0)
    stdall = np.std(y_true, axis=0)
    
    autocorrelation = np.empty(outdims)+np.nan
    persistence_MSE = np.empty(outdims)+np.nan
    
    
    for i in range(outdims[0]):
        for j in range(outdims[1]):
    
            pearsonrplot[i, j], pearsonpplot[i, j] = pearsonr(
            y_pred[:, i, j], y_out[:, i, j])
            
            autoflat_true = y_auto[:,(leadtime+run):,i,j]
            autoflat_true = np.reshape(autoflat_true,((nyears-(leadtime+run))*nvars))
            
            autoflat_pred = y_auto[:,:-1*(leadtime+run),i,j]
            autoflat_pred = np.reshape(autoflat_pred,((nyears-(leadtime+run))*nvars))
            
            autocorrelation[i,j],_ = pearsonr(autoflat_pred,autoflat_true)
            persistence_MSE[i,j] = np.mean((autoflat_pred-autoflat_true)**2)

    meanerr = np.mean(y_pred-y_true,axis=0)    
    
    plt.figure(figsize=(24, 14))
    
    a1 = plt.subplot(3, 2, 1, projection=projection)
    c1 = a1.pcolormesh(lon, lat, mseplot, transform=transform,
    cmap=cmr.fall_r, vmin=0.4, vmax=0.9)
    a1.coastlines(color='gray')
    plt.colorbar(c1)
    plt.title('MSE')
    
    a2 = plt.subplot(3, 2, 2, projection=projection)
    c2 = a2.pcolormesh(lon, lat, pearsonrplot,
    transform=transform, cmap='RdBu_r', vmin=-1, vmax=1)
    a2.coastlines(color='gray')
    plt.colorbar(c2)
    plt.title('Correlation Coefficient')
    
    a3 = plt.subplot(3, 2, 3, projection=projection)
    c3 = a3.pcolormesh(lon, lat, stdpred/stdall, transform=transform,
    cmap=cmr.fall, vmin=0, vmax=1)
    a3.coastlines(color='gray')
    plt.colorbar(c3)
    plt.title('var(y_pred)/var(y_true)')
    
    a4 = plt.subplot(3, 2, 4, projection=projection)
    c4 = a4.pcolormesh(lon, lat, meanerr,
    transform=transform, cmap=cmr.fusion_r, vmin=-1, vmax=1)
    a4.coastlines(color='gray')
    plt.colorbar(c4)
    plt.title('mean error (not absolute error)')
    
    a5 = plt.subplot(3, 2, 5, projection=projection)
    c5 = a5.pcolormesh(lon, lat, persistence_MSE, transform=transform,
    cmap=cmr.fall_r, vmin=0.4, vmax=0.9)
    a5.coastlines(color='gray')
    plt.colorbar(c5)
    plt.title('persistence MSE')
    
    a6 = plt.subplot(3, 2, 6, projection=projection)
    c6 = a6.pcolormesh(lon, lat, autocorrelation, transform=transform,
    cmap='RdBu_r', vmin=-1, vmax=1)
    a6.coastlines(color='gray')
    plt.colorbar(c6)
    plt.title('persistence correlation')
    
    plt.suptitle(title)
    
    plt.tight_layout()
    
    # plt.savefig("figures/mapmetrics_"+title+".png",dpi=300)
    
    plt.show()

def metrics(y_pred,y_true):
     
    y_out = np.copy(y_true)
     
    mse = np.mean((y_pred-y_out)**2, axis=0)
    outdims = y_pred.shape[1:]
    
    pearsonrplot = np.empty(outdims)+np.nan
    pearsonpplot = np.empty(outdims)+np.nan
    
    for i in range(outdims[0]):
        for j in range(outdims[1]):
    
            pearsonrplot[i, j], pearsonpplot[i, j] = pearsonr(
            y_pred[:, i, j], y_out[:, i, j])

    return pearsonrplot,mse










