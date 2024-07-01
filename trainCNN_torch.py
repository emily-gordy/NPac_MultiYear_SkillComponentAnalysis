#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:01:43 2024

@author: egordon4
"""
#%%
import numpy as np
import glob
# import tensorflow as tf

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import matplotlib as mpl

import random
import time

import importlib as imp
import sys
sys.path.append("functions/")

import preprocessing
import experiment_settings
import build_model

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

modelpath = "models/"
experiment_name = "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

filefront = experiment_dict["filename"]
filename = modelpath + experiment_dict["filename"]

trainvariants = experiment_dict["trainvariants"]
valvariants = experiment_dict["valvariants"]
testvariants = experiment_dict["testvariants"]
trainvaltest = [trainvariants,valvariants,testvariants]

datafile = "processed_data/" + filefront + ".npz"
filecheck = glob.glob(datafile)
saveflag = len(filecheck)==0

if saveflag:
    allinputdata,alloutputdata = preprocessing.make_inputoutput_modellist_flexavg(experiment_dict)
    
    print('got data, converting to usable format')
    
    allinputdata = np.asarray(allinputdata)
    alloutputdata = np.asarray(alloutputdata)
    
    np.savez(datafile,
             allinputdata = allinputdata,
             alloutputdata = alloutputdata)
    
else:
    datamat = np.load(datafile)
    
    allinputdata = datamat["allinputdata"]
    alloutputdata = datamat["alloutputdata"]

inputdata,inputval,inputtest,outputdata,outputval,outputtest = preprocessing.splitandflatten_torch(
    allinputdata,alloutputdata,trainvaltest,experiment_dict["inputrun"])

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

#%%

patience = experiment_dict["patience"]
batch_size = experiment_dict["batch_size"]
seedlist = experiment_dict["seeds"]
lr_patience = experiment_dict["lr_patience"]

modellist = experiment_dict["modellist"]
outbounds = experiment_dict["outbounds"]

print(outbounds)

lon, lat = preprocessing.outlonxlat(experiment_dict)
nvars = int(len(valvariants)*len(modellist))

landmask = (np.mean(outputval,axis=0))!=0
centre = (outbounds[2]+outbounds[3])/2
latweights = np.sqrt(np.cos(np.deg2rad(np.meshgrid(lon,lat)[1])))
latweights[~landmask] = 0
latweights = torch.tensor(latweights)

nvariant = len(valvariants)
nmodels = len(modellist)
ntimesteps = int(len(outputval)/(nvariant*nmodels))

inputdata_tensor = torch.tensor(inputdata)
inputval_tensor = torch.tensor(inputval)
outputdata_tensor = torch.tensor(outputdata)
outputval_tensor = torch.tensor(outputval)

input_shape = inputdata.shape
output_shape = outputdata.shape

train_dataset = TensorDataset(inputdata_tensor, outputdata_tensor)
train_loader = DataLoader(train_dataset, batch_size=experiment_dict["batch_size"], 
# latweights[~landmask] = 0
                          shuffle=True,num_workers=0)

val_dataset = TensorDataset(inputval_tensor, outputval_tensor)
val_loader = DataLoader(val_dataset, batch_size=experiment_dict["batch_size"], shuffle=False)

#%% NN functions

class weightedMSE(nn.Module):
    def __init__(self,weights):
        super(weightedMSE, self).__init__()
        self.weights=weights

    def forward(self, inputs, targets):
        loss = ((targets-inputs)*self.weights)**2
        return loss.mean()


def train_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


def val_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()

    valid_loss /= num_batches
    print(f"validation loss: {valid_loss:>7f}")

    scheduler.step(valid_loss)
    after_lr = optimizer.param_groups[0]["lr"]
    print(f"learning rate: {after_lr:>7f}")

    return valid_loss

def model_checkpoint(val_loss,best_val_loss,epochs_no_improve,fileout):

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Loss improved, saving model to "+ fileout)
        torch.save(full_model.state_dict(), fileout)
        epochs_no_improve = 0
        earlystopping=0
    else:
        epochs_no_improve += 1
        print("Loss did not improve")
        if epochs_no_improve == patience:
            earlystopping=1
        else:
            earlystopping=0
    
    return best_val_loss, earlystopping, epochs_no_improve

loss_fn = weightedMSE(latweights)

#%% training CNNs

for random_seed in seedlist:

    fileout = filename + "_seed=" + str(random_seed) +".pt"

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    full_model = build_model.CNN(input_shape, output_shape, 
                                                experiment_dict)   

    optimizer = optim.SGD(full_model.parameters(), 
                        lr=experiment_dict["learning_rate"],
                        # weight_decay=experiment_dict["ridgepen"],
                        )

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=lr_patience, cooldown=0, min_lr=5e-6, verbose=True) 

    best_val_loss = np.inf
    epochs_no_improve = 0

    epochs = experiment_dict["n_epochs"]

    validvec = []
    trainvec  = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        time1 = time.time()
        train_loss = train_loop(train_loader, full_model, loss_fn)
        valid_loss = val_loop(val_loader, full_model, loss_fn)
        trainvec.append(train_loss.detach().numpy())
        validvec.append(valid_loss)
        time2 = time.time()
        print(f"{time2-time1:4f} seconds per epoch")
        best_val_loss, earlystopping, epochs_no_improve = model_checkpoint(valid_loss,best_val_loss,epochs_no_improve,fileout)
        if earlystopping==1:
            print(f'Early stopping after {t+1} epochs.')
            break

    
    # plt.figure(figsize=(5,3))
    # plt.plot(trainvec)
    # plt.plot(validvec)
    # plt.xlabel('epoch')
    # plt.ylabel('mse')
    # plt.show()
# %%
