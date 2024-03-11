#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:58:16 2024

@author: egordon4
"""

import numpy as np
import pickle as pkl

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
experiment_name = "allcmodel-tos_allcmodel-tos_1-5yearlead_tvtfolds"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

filefront = experiment_dict["filename"]
filename = modelpath + experiment_dict["filename"]

ntrainvariants = experiment_dict["ntrainvariants"]
nvalvariants = experiment_dict["nvalvariants"]
ntestvariants = experiment_dict["ntestvariants"]

foldseeds = experiment_dict["foldseeds"]

#%%

data_experiment_name = "allcmodel-tos_allcmodel-tos_1-5yearlead"
data_experiment_dict = experiment_settings.get_experiment_settings(data_experiment_name)
datafilefront = data_experiment_dict["filename"]
datafile = "../processed_data/" + datafilefront + ".npz"

datamat = np.load(datafile)

allinputdata = datamat["allinputdata"]
alloutputdata = datamat["alloutputdata"] 


#%% functions for training

def weightedMSE(weights):
    weights = tf.cast(weights,tf.float32)
    def MSE(y_pred,y_true):
        err = tf.math.multiply((y_pred-y_true),weights)
        sqerr = tf.math.square(err)
        mse = tf.math.reduce_mean(sqerr)
        return mse
    return MSE

def scheduler(epoch, lr):
  if epoch < 20:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

#%% some useful things

patience = experiment_dict["patience"]
seedlist = experiment_dict["seeds"]

modellist = experiment_dict["modellist"]
outbounds = experiment_dict["outbounds"]

lon, lat = preprocessing.outlonxlat(experiment_dict)
nvars = int(nvalvariants*len(modellist))

centre = (outbounds[2]+outbounds[3])/2
latweights = np.sqrt(np.cos(np.deg2rad(np.meshgrid(lon,lat)[1])))


lr_callback = tf.keras.callbacks.LearningRateScheduler(
                scheduler, verbose=0
                )

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=patience, restore_best_weights=True)  # early stopping

nvariant = nvalvariants
nmodels = len(modellist)

#%% training time

trainvaltestmat = []
nmems = 30

for ifold, foldseed in enumerate(foldseeds):
    
    np.random.seed(foldseed)
    memorder = np.random.choice(np.arange(nmems),nmems,replace=False)
    
    trainvaltest = [
                    memorder[:ntrainvariants],
                    memorder[ntrainvariants:(ntrainvariants+nvalvariants)],
                    memorder[(ntrainvariants+nvalvariants):]
                    ]
    
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
    
    trainvaltestmat.append(memorder)
    
    ntimesteps = int(len(outputval)/(nvariant*nmodels))
    landmask = (np.mean(outputval,axis=0))!=0

    for random_seed in seedlist:
    
        fileout = filename + "_seed=" + str(random_seed) + "_foldseed_" + str(foldseed) +".h5"
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(fileout, 
                            monitor="val_loss", mode="min", 
                            save_best_only=True, 
                            save_weights_only=True,
                            verbose=1)
        
        callbacks = [es_callback,
                     checkpoint,
                     lr_callback,
                     ]
        
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        full_model = build_model.build_CNN_full(inputdata, outputdata, 
                                                        experiment_dict, random_seed)   
        
        full_model.compile(optimizer=tf.keras.optimizers.legacy.SGD(experiment_dict["learning_rate"]),  # optimizer
                           loss=weightedMSE(latweights),   # loss function
                           metrics=[tf.keras.metrics.MeanAbsoluteError()]
                           )
        
        full_model.summary()
        
        history = full_model.fit(x=inputdata,
                                  y=outputdata,
                                  batch_size=experiment_dict["batch_size"],
                                  epochs=experiment_dict["n_epochs"],
                                  verbose=2,
                                  callbacks=callbacks,
                                  validation_data=(inputval, outputval),
                                  shuffle=True,)
    
        full_model.load_weights(fileout)
        
        full_model.trainable = False # freeze BN
        
        metricplots.historyplot(history)
        
        y_pred = full_model.predict(inputval)
        
        title = "seed = " + str(random_seed) + " ifold = " + str(ifold)
        metricplots.mapmetrics(y_pred, outputval, nvars, lon, lat, centre, title, experiment_dict)
            
        bestpattern = allthelinalg.calculate_SC(y_pred,outputval,landmask)  
        analysisplots.plotpattern(bestpattern,lon,lat)

#%%

trainvaltestfile = "../processed_data/foldseeds" + filefront + ".pkl"

with open(trainvaltestfile,'wb') as f:
    pkl.dump(trainvaltestmat, f)


