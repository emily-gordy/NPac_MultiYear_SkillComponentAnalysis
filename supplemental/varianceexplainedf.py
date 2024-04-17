#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:10:17 2024

@author: egordon4
"""

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmasher as cmr

import sys
sys.path.append("../functions/")

import preprocessing
import experiment_settings
import build_model
import metricplots
import allthelinalg
import analysisplots
import LIM

from scipy.stats import pearsonr
from scipy.linalg import eig
from scipy.stats import ttest_ind

import pylab


# pretty plots
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.sans-serif']=['Verdana']

params = {"ytick.color": "k",
          "xtick.color": "k",
          "axes.labelcolor": "k",
          "axes.edgecolor": "k"}
plt.rcParams.update(params)

# %% load and preprocess data

modelpath = "../models/"
experiment_name = "allcmodel-tos_allcmodel-tos_1-5yearlead"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

filefront = experiment_dict["filename"]
filename = modelpath + experiment_dict["filename"]

trainvariants = experiment_dict["trainvariants"]
valvariants = experiment_dict["valvariants"]
testvariants = experiment_dict["testvariants"]
trainvaltest = [trainvariants,valvariants,testvariants]
run = experiment_dict["run"]
outbounds = experiment_dict["outbounds"]
leadtime = experiment_dict["leadtime"]
seedlist = experiment_dict["seeds"]


obsyearvec = np.arange(1870+3*run+leadtime,2023,)

modellist = experiment_dict["modellist"]

datafile = "../processed_data/" + filefront + ".npz"

datamat = np.load(datafile)

allinputdata = datamat["allinputdata"]
alloutputdata = datamat["alloutputdata"]

inputdata,inputval,inputtest,outputdata,outputval,outputtest = preprocessing.splitandflatten(
    allinputdata,alloutputdata,trainvaltest,experiment_dict["run"])

inputdata[:, np.isnan(np.mean(inputdata, axis=0))] = 0
inputval[:, np.isnan(np.mean(inputval, axis=0))] = 0
inputtest[:, np.isnan(np.mean(inputtest, axis=0))] = 0

outputstd = np.std(outputdata, axis=0, keepdims=True)
outputdata = outputdata/outputstd
outputval = outputval/outputstd
outputtest = outputtest/outputstd

outputdata[:, np.isnan(np.mean(outputdata, axis=0))] = 0
outputval[:, np.isnan(np.mean(outputval, axis=0))] = 0
outputtest[:, np.isnan(np.mean(outputtest, axis=0))] = 0  

lon, lat = preprocessing.outlonxlat(experiment_dict)
landmask = (np.mean(outputval,axis=0))!=0

nvalvars = int(len(valvariants)*len(modellist))
ntestvars = int(len(valvariants)*len(modellist))

centre = (outbounds[2]+outbounds[3])/2

inputobs_ERSST,outputobs_ERSST = preprocessing.make_inputoutput_obs(experiment_dict,"ERSST")
inputobs_ERSST,outputobs_ERSST = preprocessing.concatobs(inputobs_ERSST,outputobs_ERSST,outputstd,run)

inputobs_HadISST,outputobs_HadISST = preprocessing.make_inputoutput_obs(experiment_dict,"HadISST")
inputobs_HadISST,outputobs_HadISST = preprocessing.concatobs(inputobs_HadISST,outputobs_HadISST,outputstd,run)

#%%

random_seed = seedlist[3]

fileout = filename + "_seed=" + str(random_seed) +".h5"

tf.random.set_seed(random_seed)
np.random.seed(random_seed) 

full_model = build_model.build_CNN_full(inputdata, outputdata, 
                                                experiment_dict, random_seed)  

full_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(experiment_dict["learning_rate"]),  # optimizer
                    loss=tf.keras.losses.MeanSquaredError(), 
                  )

full_model.load_weights(fileout)

full_model.trainable = False # freeze BN

y_pred_val_CNN = full_model.predict(inputval) 
y_pred_test_CNN = full_model.predict(inputtest)

r_test_CNN,mse_test_CNN = metricplots.metrics(y_pred_test_CNN,outputtest)

y_pred_CNN_ERSST = full_model.predict(inputobs_ERSST)
y_pred_CNN_HadISST = full_model.predict(inputobs_HadISST)

SC_CNN = allthelinalg.calculate_SC(y_pred_val_CNN,outputval,landmask)
SC_CNN_test_timeseries = allthelinalg.index_timeseries(outputtest,SC_CNN,landmask)
SC_CNN_ERSST_timeseries = allthelinalg.index_timeseries(outputobs_ERSST,SC_CNN,landmask)

#%%

def get_variance_explained(SC_index,data):
    
    datashape = data.shape
    
    SC_variance_explained = np.empty((datashape[1],datashape[2]))
    
    for ilat in range(datashape[1]):
        for ilon in range(datashape[2]):
            SC_variance_explained[ilat,ilon],_ = pearsonr(SC_index,data[:,ilat,ilon])
    
    SC_variance_explained = SC_variance_explained**2
    
    return SC_variance_explained
    

#%%

varexplained_test_SST = get_variance_explained(SC_CNN_test_timeseries,outputtest)
varexplained_ERSST = get_variance_explained(SC_CNN_ERSST_timeseries[:-5],inputobs_ERSST[5:,:,:,0])

#%%

projection = ccrs.PlateCarree(central_longitude=180)
transform = ccrs.PlateCarree()

plt.figure(figsize=(10,5))

a1=plt.subplot(1,2,1,projection=projection)
a1.pcolormesh(lon,lat,varexplained_test_SST,vmin=0,vmax=1,transform=transform,cmap="inferno")
a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))

a2=plt.subplot(1,2,2,projection=projection)
a2.pcolormesh(lon,lat,varexplained_ERSST,vmin=0,vmax=1,transform=transform,cmap="inferno")
a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))

plt.tight_layout()
plt.show()

# now do this with surface temp? or just show how great we are at marine heatwaves?









