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
experiment_name = "allcmodel-tos_allcmodel-tos_1-3yearlead"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

filefront = experiment_dict["filename"]
filename = modelpath + experiment_dict["filename"]

trainvariants = experiment_dict["trainvariants"]
valvariants = experiment_dict["valvariants"]
testvariants = experiment_dict["testvariants"]
trainvaltest = [trainvariants,valvariants,testvariants]
run = experiment_dict["run"]

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

def reshapengrab(mat,imodel,nvariants):
    
    inputshape = mat.shape
    ntimesteps = int(inputshape[0]/(nvariants*nmodels))
    if len(inputshape) == 3:        
        matreshape = np.reshape(mat,(nmodels,nvariants,ntimesteps,inputshape[1],inputshape[2]))
    else:
        matreshape = np.reshape(mat,(nmodels,nvariants,ntimesteps,inputshape[1],inputshape[2],inputshape[3]))
    
    matgrab = matreshape[imodel]
    
    if len(inputshape) == 3:
        shapeout = (ntimesteps*nvariants,inputshape[1],inputshape[2])
    else:
        shapeout = (ntimesteps*nvariants,inputshape[1],inputshape[2],inputshape[3])
    matout = np.reshape(matgrab,shapeout)    

    return matout


NUM_COLORS = len(lat)
cvec = []
cm = pylab.get_cmap('cmr.eclipse')
for i in range(NUM_COLORS):
    color = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple
    cvec.append(color)

#%%

def haversine(lon1,lat1,lon2,lat2):
    
    d_r = 2*np.arcsin(np.sqrt((np.sin(np.deg2rad((lat2-lat1)/2)))**2 + 
                      np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(lat1))*(np.sin(np.deg2rad(lon2-lon1)/2))**2)
                      )
    
    return d_r

longrid,latgrid = np.meshgrid(lon,lat)

nvars = len(valvariants)
nmodels = len(modellist)
ntimesteps = int(len(outputval)/(nvars*nmodels))

landmask = (np.mean(outputval,axis=0))!=0

latflat = latgrid[landmask]
lonflat = longrid[landmask]

npoints = len(latflat)
def variances(data):
    
    dataflat = data[:,landmask]
    
    # dists = np.empty((npoints,(npoints-1)))
    # corr = np.empty((npoints,(npoints-1)))
    
    alldist = []
    allcorr = []
    
    for ii in range(npoints):
    
        subjvec = dataflat[:,ii]        
        objvecs = dataflat[:,(ii+1):]
        
        ndists = objvecs.shape[1]
        
        dists = np.empty((ndists))
        corrs = np.empty((ndists))
        
        for point2 in range(ndists):    
        
            dists[point2] = haversine(lonflat[ii],latflat[ii],lonflat[ii+point2],latflat[ii+point2])
            corrs[point2],_ = pearsonr(subjvec,objvecs[:,point2])
            # corrs[point2] = np.mean((subjvec-objvecs[:,point2])**2)
            
        alldist.append(dists)
        allcorr.append(corrs)
            
    return alldist,allcorr
    
    
#%%

plt.figure(figsize=(18,16))

for imodel,cmodel in enumerate(modellist):
    
    datagrab = reshapengrab(outputval,imodel,len(valvariants))
        
    d,c = variances(datagrab)
    
    plt.subplot(3,3,imodel+1)
    
    for iplot in range(npoints):
        
        latplot = latflat[iplot]
        latind = np.where(lat==latplot)[0][0]
        cplot = cvec[latind]

        plt.scatter(d[iplot],c[iplot],marker='.',color=cplot)
    
    plt.xlabel(r"distance ($R_{E}$)")
    plt.ylabel("mean squared difference")
    plt.ylim(0,7)
    plt.title(cmodel)
    
plt.tight_layout()
plt.show()

#%%

sources = ["ERSST","HadISST"]

plt.figure(figsize=(8,4))

for isource,source in enumerate(sources):

    inputobs,outputobs = preprocessing.make_inputoutput_obs(experiment_dict,source)
    inputobs,outputobs = preprocessing.concatobs(inputobs,outputobs,outputstd,run)
        
    d,c = variances(outputobs)
    
    for iplot in range(npoints):
        
        latplot = latflat[iplot]
        latind = np.where(lat==latplot)[0][0]
        cplot = cvec[latind]
        
        plt.subplot(1,2,isource+1)
        plt.scatter(d[iplot],c[iplot],marker='.',color=cplot)
    
    plt.xlabel(r"distance ($R_{E}$)")
    plt.ylabel("correlation coefficient")
    plt.title(source)
    plt.ylim(-1,1)

plt.tight_layout()
plt.show()

#%%

Rlims = np.arange(0,0.5,0.1)

plt.figure(figsize=(10,8))

for ilim,rlim in enumerate(Rlims[:-1]):

    plt.subplot(2,2,ilim+1)    

    for iplot in range(npoints):
        
        latplot = latflat[iplot]
        latind = np.where(lat==latplot)[0][0]
        colorplot = cvec[latind]
        
        lonplot = lonflat[iplot]
        
        cloop = c[iplot]
        dloop = d[iplot]
        
        Rboo = (dloop>=Rlims[ilim]) & (dloop<Rlims[ilim+1])
        
        cplot = cloop[Rboo]
        
        latplotvec = latplot*np.ones(len(cplot))
        lonplotvec = lonplot*np.ones(len(cplot))
        
        c1=plt.scatter(latplotvec,cplot,c=lonplotvec,cmap="inferno",vmin=np.min(lon),vmax=np.max(lon))
    
    cbar=plt.colorbar(c1)    
    cbar.ax.set_ylabel("longitude")
    plt.title(str(rlim) + r" $R_{E}$ < d < " + str(rlim+0.1) + r" $R_{E}$")
    plt.xlabel("latitude")
    plt.ylabel("correlation coefficient")


plt.tight_layout()
plt.show()




















