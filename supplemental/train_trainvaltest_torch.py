#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:58:16 2024

@author: egordon4
"""

import numpy as np
import pickle as pkl
import random
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import sys
import os

path = "/Users/egordon4/Documents/Experiments/NPac_MultiYear_SkillComponentAnalysis/"
sys.path.append(path + "functions")

import preprocessing
import experiment_settings
import build_model


# load and preprocess data

modelpath = path + "/models/"
experiment_name = "allcmodel-tos_allcmodel-tos_2-9yearlead_flexvg_tvtfolds"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

filefront = experiment_dict["filename"]
filename = modelpath + experiment_dict["filename"]

ntrainvariants = experiment_dict["ntrainvariants"]
nvalvariants = experiment_dict["nvalvariants"]
ntestvariants = experiment_dict["ntestvariants"]

foldseeds = experiment_dict["foldseeds"]

data_experiment_name = "allcmodel-tos_allcmodel-tos_2-9yearlend_flexavg"
data_experiment_dict = experiment_settings.get_experiment_settings(data_experiment_name)
datafilefront = data_experiment_dict["filename"]
datafile = path + "/processed_data/" + datafilefront + ".npz"

datamat = np.load(datafile)

allinputdata = datamat["allinputdata"]
alloutputdata = datamat["alloutputdata"] 

patience = experiment_dict["patience"]
seedlist = experiment_dict["seeds"]
modellist = experiment_dict["modellist"]
outbounds = experiment_dict["outbounds"]
lr_patience = experiment_dict["lr_patience"]
batch_size = experiment_dict["batch_size"]
epochs =  experiment_dict["n_epochs"]

lon, lat = preprocessing.outlonxlat(experiment_dict)
nvars = int(nvalvariants*len(modellist))

centre = (outbounds[2]+outbounds[3])/2
latweights = np.sqrt(np.cos(np.deg2rad(np.meshgrid(lon,lat)[1])))

#%% NN funcs

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

nvariant = nvalvariants
nmodels = len(modellist)

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
    
    trainvaltestmat.append(memorder)
    
    ntimesteps = int(len(outputval)/(nvariant*nmodels))
    landmask = (np.mean(outputval,axis=0))!=0

    inputdata_tensor = torch.tensor(inputdata)
    inputval_tensor = torch.tensor(inputval)
    outputdata_tensor = torch.tensor(outputdata)
    outputval_tensor = torch.tensor(outputval)

    train_dataset = TensorDataset(inputdata_tensor, outputdata_tensor)
    train_loader = DataLoader(train_dataset, batch_size=experiment_dict["batch_size"], 
                          shuffle=True)

    val_dataset = TensorDataset(inputval_tensor, outputval_tensor)
    val_loader = DataLoader(val_dataset, batch_size=experiment_dict["batch_size"], 
                          shuffle=False)

    input_shape = inputdata.shape
    output_shape = outputdata.shape

    for random_seed in seedlist:
    
        fileout = filename + "_seed=" + str(random_seed) + "_foldseed_" + str(foldseed) +".pt"
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        full_model = build_model.CNN(input_shape, output_shape, 
                                        experiment_dict)   

        optimizer = optim.SGD(full_model.parameters(), 
                        lr=experiment_dict["learning_rate"],
                        weight_decay=experiment_dict["ridgepen"])

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=lr_patience, cooldown=0, min_lr=5e-6, verbose=True) 

        loss = []

        best_val_loss = np.inf
        epochs_no_improve = 0

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            time1 = time.time()
            train_loop(train_loader, full_model, loss_fn)
            valid_loss = val_loop(val_loader, full_model, loss_fn)
            loss.append(valid_loss)
            time2 = time.time()
            print(f"{time2-time1:4f} seconds per epoch")
            best_val_loss, earlystopping, epochs_no_improve = model_checkpoint(valid_loss,best_val_loss,epochs_no_improve,fileout)
            if earlystopping==1:
                print(f'Early stopping after {t+1} epochs.')
                break

trainvaltestfile = path + "/processed_data/foldseeds" + filefront + ".pkl"

with open(trainvaltestfile,'wb') as f:
    pkl.dump(trainvaltestmat, f)


