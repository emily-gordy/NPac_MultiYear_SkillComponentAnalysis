#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:54:24 2024

@author: egordon4
"""

import numpy as np
from scipy.linalg import eig
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import allthelinalg
import matplotlib.pyplot as plt
import cmasher as cmr

def LIM_getG(data,tau,dims,landmask,weights):
    
    data = data*weights[np.newaxis,:,:]
    
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

def get_normalmodes(G,landmask):
    
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
    
    
    
    
    
    