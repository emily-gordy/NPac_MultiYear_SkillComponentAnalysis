#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:10:09 2024

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

#%%

inputobs_ERSST,outputobs_ERSST = preprocessing.make_inputoutput_obs(experiment_dict,"ERSST")
inputobs_ERSST,outputobs_ERSST = preprocessing.concatobs(inputobs_ERSST,outputobs_ERSST,outputstd,run)

inputobs_HadISST,outputobs_HadISST = preprocessing.make_inputoutput_obs(experiment_dict,"HadISST")
inputobs_HadISST,outputobs_HadISST = preprocessing.concatobs(inputobs_HadISST,outputobs_HadISST,outputstd,run)


#%%

tau = 5

weights = np.meshgrid(lon,lat)[1]
latweights = np.sqrt(np.cos(np.deg2rad(weights)))

bigdatashape = alloutputdata.shape

traindims = LIM.get_partial_dims(alloutputdata,trainvariants)
valdims = LIM.get_partial_dims(alloutputdata,valvariants)
testdims = LIM.get_partial_dims(alloutputdata,testvariants)

G = LIM.LIM_getG(outputdata,tau,traindims,landmask,weights)

y_pred_val_LIM,outputval_tau = LIM.LIM_cmodel(outputval,tau,valdims,landmask,G)
y_pred_test_LIM,outputtest_tau = LIM.LIM_cmodel(outputtest,tau,testdims,landmask,G)

r_test_LIM,mse_test_LIM = metricplots.metrics(y_pred_test_LIM,outputtest_tau)

y_pred_LIM_ERSST,outputobs_ERSST_tau = LIM.LIM_obs(outputobs_ERSST,tau,landmask,G)
y_pred_LIM_HadISST,outputobs_HadISST_tau = LIM.LIM_obs(outputobs_HadISST,tau,landmask,G)

r_ERSST_LIM,mse_ERSST_LIM = metricplots.metrics(y_pred_LIM_ERSST,outputobs_ERSST_tau)

mse_ERSST_LIM_full = np.nanmean((y_pred_LIM_ERSST-outputobs_ERSST_tau)**2)

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

r_ERSST_CNN,mse_ERSST_CNN = metricplots.metrics(y_pred_CNN_ERSST,outputobs_ERSST)

mse_ERSST_CNN_full = np.nanmean((y_pred_CNN_ERSST-outputobs_ERSST)**2)

#%%

mse_ERSST_CNN[np.isnan(mse_ERSST_CNN)] = 0
mse_ERSST_LIM[np.isnan(mse_ERSST_LIM)] = 0
r_ERSST_CNN[np.isnan(r_ERSST_CNN)] = 0
r_ERSST_LIM[np.isnan(r_ERSST_LIM)] = 0


#%%

plt.figure(figsize=(8,5))

a1=plt.subplot(2,1,1,projection=ccrs.PlateCarree(central_longitude=180))
c1=a1.pcolormesh(lon,lat,-1*(mse_ERSST_CNN-mse_ERSST_LIM),vmin=-0.5,vmax=0.5,cmap=cmr.redshift,transform=ccrs.PlateCarree())
a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))

a2=plt.subplot(2,1,2,projection=ccrs.PlateCarree(central_longitude=180))
c2=a2.pcolormesh(lon,lat,r_ERSST_CNN-r_ERSST_LIM,vmin=-0.5,vmax=0.5,cmap=cmr.redshift,transform=ccrs.PlateCarree())
a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))

cax=plt.axes((0.9,0.2,0.02,0.6))
cbar = plt.colorbar(c1,cax=cax,extend='both')
cbar.ax.set_yticks(np.arange(-0.5,1,0.5))

#%%

nmode = LIM.get_normalmodes(G,landmask)
SC_LIM = allthelinalg.calculate_SC(y_pred_val_LIM,outputval_tau,landmask)
SC_CNN = allthelinalg.calculate_SC(y_pred_val_CNN,outputval,landmask)

nmodetrue = allthelinalg.index_timeseries(outputobs_ERSST_tau,nmode,landmask)
nmodepred = allthelinalg.index_timeseries(y_pred_LIM_ERSST,nmode,landmask)

SC_LIM_true = allthelinalg.index_timeseries(outputobs_ERSST_tau,SC_LIM,landmask)
SC_LIM_pred = allthelinalg.index_timeseries(y_pred_LIM_ERSST,SC_LIM,landmask)

SC_CNN_true = allthelinalg.index_timeseries(outputobs_ERSST,SC_CNN,landmask)
SC_CNN_pred = allthelinalg.index_timeseries(y_pred_CNN_ERSST,SC_CNN,landmask)

#%%

plt.figure(figsize=(10,4))

plt.plot(obsyearvec,nmodetrue,color="xkcd:black",
         label="LIM Normal Mode",linewidth=1.8)
plt.plot(obsyearvec,SC_CNN_true[5:],color='xkcd:slate',
         label="CNN Skill Component",linewidth=1.8)
plt.plot(obsyearvec,nmodepred,color="xkcd:golden rod",linestyle='--',
         label="LIM Predicted Normal Mode",linewidth=2.2)
plt.plot(obsyearvec,SC_CNN_pred[5:],color='xkcd:teal',
         linestyle='-.',label="CNN Predicted Skill Component",linewidth=2.2)

plt.legend()

r_LIM,_ = pearsonr(nmodetrue,nmodepred)
r_CNN,_ = pearsonr(SC_CNN_true,SC_CNN_pred)

plt.text(1995,-1.9,"r = %.4f" %(r_LIM),color='xkcd:golden rod',fontsize=16,weight='bold')
plt.text(1995,-2.3,"r = %.4f" %(r_CNN),color='xkcd:teal',fontsize=16,weight='bold')

plt.xlabel('year')
plt.ylabel('index')

plt.ylim(-2.8,2.8)
plt.xlim(obsyearvec[0],obsyearvec[-1])

plt.tight_layout()

# plt.savefig("../figures/LIMcomparison.png",dpi=300)

plt.show()
