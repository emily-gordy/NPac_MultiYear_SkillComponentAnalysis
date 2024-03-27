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
    
    y_pred = np.empty((y_pred_LIM.shape[0],dims[1],dims[2]))+np.nan
    y_pred[:,landmask] = y_pred_LIM
    
    return y_pred,data_tau

def get_partial_dims(data,variants):
    
    dimsout = (data.shape[0],len(variants),data.shape[2],data.shape[3],data.shape[4])
    
    return dimsout

def get_normalmodes(G):
    
    eigvals,evecs = eig(G)
    bestinds = np.argsort(np.real(eigvals))
    
    ivec = -1
    
    evecsel = evecs[:,bestinds[ivec]]
    
    if np.sum(evecsel)<0:
        evecsel = -1*evecsel
    
    nmode = np.zeros((11,38))+np.nan
    nmode[landmask] = np.real(evecsel)
    
    return nmode

def patternplots_SST(bestpattern,PDOpattern,truedata,preddata,outputval,y_pred_val,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SCstd_true,SCstd_pred = allthelinalg.standardizations(outputval,y_pred_val,bestpattern,landmask)
    PDOstd_true,PDOstd_pred = allthelinalg.standardizations(outputval,y_pred_val,PDOpattern,landmask)
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)#/SCstd_true
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)#/SCstd_pred
    
    PDOindex_true = allthelinalg.index_timeseries(truedata,PDOpattern,landmask)#/PDOstd_true
    PDOindex_pred = allthelinalg.index_timeseries(preddata,PDOpattern,landmask)#/PDOstd_pred
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    PDOpatternplot = np.mean(truedata*PDOindex_true[:,np.newaxis,np.newaxis],axis=0)
    
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    PDOpatternplot = PDOpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    cc_PDO,_ = pearsonr(PDOindex_true,PDOindex_pred)
    
    plt.figure(figsize=(19,9))
    
    a1=plt.subplot(2,3,1,projection=projection)
    a1.pcolormesh(lon,lat,bestpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar1.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title('SC pattern (LIM)')
    
    plt.subplot(2,3,2)
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:teal')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true SC')
    plt.ylabel('pred SC')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')
    
    plt.subplot(2,3,3)
    plt.plot(yearvec,SC_index_true,color='#002fa7',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:golden rod',label='pred')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('SC index')
    plt.xlabel('year')

    a2=plt.subplot(2,3,4,projection=projection)
    a2.pcolormesh(lon,lat,PDOpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c2=a2.contourf(lon,lat,PDOpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    cbar2=plt.colorbar(c2,location='bottom')
    cbar2.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar2.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title('Normal Mode (LIM)')
    
    plt.subplot(2,3,5)
    c=plt.scatter(PDOindex_true,PDOindex_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:teal')
    plt.xlabel('true NM')
    plt.ylabel('pred NM')
    plt.text(-2.5,2,"r = %.4f" %(cc_PDO))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat2=plt.colorbar(c)
    cbarscat2.ax.set_ylabel('year')
    
    plt.subplot(2,3,6)
    plt.plot(yearvec,PDOindex_true,color='#002fa7',label='true')
    plt.plot(yearvec,PDOindex_pred,color='xkcd:golden rod',label='pred')
    # plt.legend()
    plt.ylabel('NM index')
    plt.xlabel('year')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    
    plt.suptitle(title,fontsize=30)

    plt.tight_layout

    #plt.savefig("figures/" +title+"_patternscatterline.png",dpi=300)
    
    plt.show()

#%%

tau = 5

bigdatashape = alloutputdata.shape

traindims = get_partial_dims(alloutputdata,trainvariants)
valdims = get_partial_dims(alloutputdata,valvariants)
testdims = get_partial_dims(alloutputdata,testvariants)

G = LIM_getG(outputdata,tau,traindims,landmask)

y_pred_val,outputval_tau = LIM_cmodel(outputval,tau,valdims,landmask,G)
y_pred_test_LIM,outputtest_tau = LIM_cmodel(outputtest,tau,testdims,landmask,G)

# r_test_LIM,mse_test_LIM = metricplots.metrics(y_pred_test_LIM,outputtest_tau)

nmode = get_normalmodes(G)

y_pred_obs_ERSST,outputobs_ERSST_tau = LIM_obs(outputobs_ERSST,tau,landmask,G)
y_pred_obs_HadISST,outputobs_HadISST_tau = LIM_obs(outputobs_HadISST,tau,landmask,G)

nmode_true_ERSST = allthelinalg.index_timeseries(outputobs_ERSST_tau,nmode,landmask)
nmode_pred_ERSST = allthelinalg.index_timeseries(y_pred_obs_ERSST,nmode,landmask)

nmode_true_HadISST = allthelinalg.index_timeseries(outputobs_HadISST_tau,nmode,landmask)
nmode_pred_HadISST = allthelinalg.index_timeseries(y_pred_obs_HadISST,nmode,landmask)

r_LIM_ERSST,_ = pearsonr(nmode_true_ERSST,nmode_pred_ERSST)
r_LIM_HadISST,_ = pearsonr(nmode_true_HadISST,nmode_pred_HadISST)

r_test_LIM,mse_test_LIM = metricplots.metrics(y_pred_obs_ERSST,outputobs_ERSST_tau)

#%%

random_seed = seedlist[0]

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

y_pred_val = full_model.predict(inputval) 
y_pred_test_CNN = full_model.predict(inputtest)

# r_test_CNN,mse_test_CNN = metricplots.metrics(y_pred_test_CNN,outputtest)

y_pred_CNN_ERSST = full_model.predict(inputobs_ERSST)
y_pred_CNN_HadISST = full_model.predict(inputobs_HadISST)

bestpattern = allthelinalg.calculate_SC(y_pred_val,outputval,landmask)   

r_test_CNN,mse_test_CNN = metricplots.metrics(y_pred_CNN_ERSST,outputobs_ERSST)

SC_true_ERSST = allthelinalg.index_timeseries(outputobs_ERSST,bestpattern,landmask)
SC_pred_ERSST = allthelinalg.index_timeseries(y_pred_CNN_ERSST,bestpattern,landmask)

SC_true_HadISST = allthelinalg.index_timeseries(outputobs_HadISST,bestpattern,landmask)
SC_pred_HadISST = allthelinalg.index_timeseries(y_pred_CNN_HadISST,bestpattern,landmask)

r_CNN_ERSST,_ = pearsonr(SC_true_ERSST,SC_pred_ERSST)
r_CNN_HadISST,_ = pearsonr(SC_true_HadISST,SC_pred_HadISST)


#%%

nmode_testing = allthelinalg.index_timeseries(outputtest_tau,nmode,landmask)
SCindex_testing = allthelinalg.index_timeseries(outputtest,bestpattern,landmask)

nmode_pred = allthelinalg.index_timeseries(y_pred_test_LIM,nmode,landmask)
SCindex_pred = allthelinalg.index_timeseries(y_pred_test_CNN,bestpattern,landmask)

lenmodel_CNN = int(len(SCindex_testing)/9)
lenmodel_LIM = int(len(nmode_testing)/9)

    
#%%

ntimesteps_CNN = len(obsyearvec)+tau
ntimesteps_LIM = len(obsyearvec)

nmodels = len(modellist)

r_CNN = np.empty((nmodels,len(testvariants)))
r_LIM = np.empty((nmodels,len(testvariants)))

plt.figure(figsize=(10,9))

for imodel,cmodel in enumerate(modellist):
    
    trueloop = SCindex_testing[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN]
    predloop = SCindex_pred[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN]
    
    err = np.abs(trueloop-predloop)
    
    plt.subplot(3,3,imodel+1)
    plt.scatter(trueloop,predloop,
                    c=err,marker='.',cmap=cmr.ember_r)
    plt.plot(np.arange(-4,5),np.arange(-4,5),color='#002fa7')
    plt.title(cmodel)
    plt.xlim(-5,5)
    plt.ylim(-5,5)    
    # r2,_ = pearsonr(nmode_testing[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN],
    #                 nmode_pred[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN])
    
    # r_CNN.append(r1)
    # r_LIM.append(r2)
    for ivariant in range(len(testvariants)):
        r_CNN[imodel,ivariant],_ = pearsonr(trueloop[ivariant*ntimesteps_CNN:
                                                   (ivariant+1)*ntimesteps_CNN],
                                          predloop[ivariant*ntimesteps_CNN:
                                                   (ivariant+1)*ntimesteps_CNN])

plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10,9))

for imodel,cmodel in enumerate(modellist):

    trueloop = nmode_testing[imodel*lenmodel_LIM:(imodel+1)*lenmodel_LIM]
    predloop = nmode_pred[imodel*lenmodel_LIM:(imodel+1)*lenmodel_LIM]
    
    err = np.abs(trueloop-predloop)
    
    plt.subplot(3,3,imodel+1)
    plt.scatter(nmode_testing[imodel*lenmodel_LIM:(imodel+1)*lenmodel_LIM],
                    nmode_pred[imodel*lenmodel_LIM:(imodel+1)*lenmodel_LIM],
                    c=err,marker='.',cmap=cmr.ember_r)
    plt.plot(np.arange(-4,5),np.arange(-4,5),color='#002fa7')
    plt.xlim(-5,5)
    plt.ylim(-5,5)    
    plt.title(cmodel)
    
    # r2,_ = pearsonr(nmode_testing[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN],
    #                 nmode_pred[imodel*lenmodel_CNN:(imodel+1)*lenmodel_CNN])
    
    # r_CNN.append(r1)
    # r_LIM.append(r2)
    for ivariant in range(len(testvariants)):
        r_LIM[imodel,ivariant],_ = pearsonr(trueloop[ivariant*ntimesteps_LIM:
                                                   (ivariant+1)*ntimesteps_LIM],
                                          predloop[ivariant*ntimesteps_LIM:
                                                   (ivariant+1)*ntimesteps_LIM])

plt.tight_layout()
plt.show()

#%%

ttest = []

meanr_CNN = np.mean(r_CNN,axis=1)
meanr_LIM = np.mean(r_LIM,axis=1)

maxr_CNN = np.max(r_CNN,axis=1)
maxr_LIM = np.max(r_LIM,axis=1)

minr_CNN = np.min(r_CNN,axis=1)
minr_LIM = np.min(r_LIM,axis=1)

uppererrs_CNN = maxr_CNN-meanr_CNN
lowererrs_CNN = meanr_CNN-minr_CNN

uppererrs_LIM = maxr_LIM-meanr_LIM
lowererrs_LIM = meanr_LIM-minr_LIM

CNNbars = np.concatenate((lowererrs_CNN[np.newaxis,:],uppererrs_CNN[np.newaxis,:]),axis=0)
LIMbars = np.concatenate((lowererrs_LIM[np.newaxis,:],uppererrs_LIM[np.newaxis,:]),axis=0)

plt.figure(figsize=(8,8))

for imodel,cmodel in enumerate(modellist):
    
    ttestloop = ttest_ind(r_CNN[imodel,:],r_LIM[imodel,:],equal_var=False)

    ttest.append(ttestloop)

    plt.errorbar(meanr_LIM[imodel],meanr_CNN[imodel],yerr=CNNbars[:,[imodel]],xerr=LIMbars[:,[imodel]],ls=None,
                 label=cmodel,marker='o')

plt.scatter(r_LIM_ERSST,r_CNN_ERSST,marker='x',color='xkcd:golden rod')
plt.scatter(r_LIM_HadISST,r_CNN_HadISST,marker='x',color='xkcd:peach')
plt.plot(np.arange(0,2),np.arange(0,2),color='xkcd:slate grey')

plt.xlabel('LIM')
plt.ylabel('CNN')

plt.legend()


#%%

    







