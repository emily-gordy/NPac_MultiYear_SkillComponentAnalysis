#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:52:23 2024

@author: egordon4
"""

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

import sys
sys.path.append("../functions/")

import preprocessing
import experiment_settings
import build_model
import metricplots
import allthelinalg
import analysisplots

from scipy.stats import pearsonr

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

modellist = experiment_dict["modellist"]

datafile = "../processed_data/" + filefront + ".npz"

datamat = np.load(datafile)

allinputdata = datamat["allinputdata"]
alloutputdata = datamat["alloutputdata"]

inputdata,inputval,inputtest,outputdata,outputval,outputtest = preprocessing.splitandflatten(
    allinputdata,alloutputdata,trainvaltest,experiment_dict["run"])

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

#%% LIM functions

def LIM_getG(data,tau,dims,landmask):

    data = np.reshape(data,dims) # reshape to time dimension
    
    data_0 = data[:,:,:-1*tau,:,:] # lead/lag by lead time
    data_tau = data[:,:,tau:,:,:]
    
    data_0 = np.reshape(data_0,(dims[0]*dims[1]*(dims[2]-tau),dims[3],dims[4])) # and shape back 
    data_tau = np.reshape(data_tau,(dims[0]*dims[1]*(dims[2]-tau),dims[3],dims[4]))
    
    data_0 = np.transpose(data_0[:,landmask]) # dimensions space x time
    data_tau = np.transpose(data_tau[:,landmask])
    
    c_0 = np.dot(data_0,data_0.T) / (data_0.shape[1] - 1) # data variance matrix
    c_tau = np.dot(data_tau,data_0.T) / (data_0.shape[1] - 1) # data covariance matrix
    
    G = np.dot(c_tau,np.linalg.pinv(c_0)) # propagator matrix
    
    return G

def LIM_cmodel(data,tau,dims,landmask,G):

    data = np.reshape(data,dims)
    
    data_0 = data[:,:,:-1*tau,:,:]
    data_tau = data[:,:,tau:,:,:]
    
    data_0 = np.reshape(data_0,(dims[0]*dims[1]*(dims[2]-tau),dims[3],dims[4]))
    data_tau = np.reshape(data_tau,(dims[0]*dims[1]*(dims[2]-tau),dims[3],dims[4]))
    
    data_0 = np.transpose(data_0[:,landmask])
    
    y_pred_LIM = np.matmul(G,data_0)
    y_pred_LIM = np.transpose(y_pred_LIM)
    
    y_pred = np.empty((y_pred_LIM.shape[0],dims[3],dims[4]))+np.nan
    y_pred[:,landmask] = y_pred_LIM
    
    return y_pred,data_tau

def LIM_obs(data,tau,landmask,G):

    data_0 = data[:-1*tau,:,:]
    data_tau = data[tau:,:,:]
    
    dims = data_0.shape
    
    data_0 = np.transpose(data_0[:,landmask])
    
    y_pred_LIM = np.matmul(G,data_0)
    y_pred_LIM = np.transpose(y_pred_LIM)
    
    y_pred = np.empty((y_pred_LIM.shape[0],dims[3],dims[4]))+np.nan
    y_pred[:,landmask] = y_pred_LIM
    
    return y_pred,data_tau

def get_partial_dims(data,variants):
    
    dimsout = (data.shape[0],len(variants),data.shape[2],data.shape[3],data.shape[4])
    
    return dimsout

#%%

tau = 5

bigdatashape = alloutputdata.shape

traindims = get_partial_dims(alloutputdata,trainvariants)
valdims = get_partial_dims(alloutputdata,valvariants)
testdims = get_partial_dims(alloutputdata,testvariants)

G = LIM_getG(outputdata,tau,traindims,landmask)

y_pred_val,outputval_tau = LIM_cmodel(outputval,tau,valdims,landmask,G)
y_pred_test,outputtest_tau = LIM_cmodel(outputtest,tau,testdims,landmask,G)

metricplots.mapmetrics(y_pred_test, outputtest_tau, nvalvars, lon, lat, centre, 'all models, vaidation', experiment_dict)
metricplots.mapmetrics(y_pred_val, outputval_tau, ntestvars, lon, lat, centre, 'all models, testing', experiment_dict)







