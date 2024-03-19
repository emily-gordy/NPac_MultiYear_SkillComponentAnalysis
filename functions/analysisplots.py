#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:51:29 2024

@author: egordon4
"""

import xarray as xr
import glob
import numpy as np

from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.linalg import eig
from scipy.stats import skew, skewtest, percentileofscore

import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import cmasher as cmr

import allthelinalg

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
    plt.title('PPDV pattern')
    
    plt.subplot(2,3,2)
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:teal')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true PPDV')
    plt.ylabel('pred PPDV')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')
    
    plt.subplot(2,3,3)
    plt.plot(yearvec,SC_index_true,color='xkcd:teal',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:golden rod',label='pred')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('PPDV index')
    plt.xlabel('year')

    a2=plt.subplot(2,3,4,projection=projection)
    a2.pcolormesh(lon,lat,PDOpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c2=a2.contourf(lon,lat,PDOpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    cbar2=plt.colorbar(c2,location='bottom')
    cbar2.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar2.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title('PDO pattern')
    
    plt.subplot(2,3,5)
    c=plt.scatter(PDOindex_true,PDOindex_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:teal')
    plt.xlabel('true PDO')
    plt.ylabel('pred PDO')
    plt.text(-2.5,2,"r = %.4f" %(cc_PDO))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat2=plt.colorbar(c)
    cbarscat2.ax.set_ylabel('year')
    
    plt.subplot(2,3,6)
    plt.plot(yearvec,PDOindex_true,color='xkcd:teal',label='true')
    plt.plot(yearvec,PDOindex_pred,color='xkcd:golden rod',label='pred')
    # plt.legend()
    plt.ylabel('PDO index')
    plt.xlabel('year')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    
    plt.suptitle(title,fontsize=30)

    plt.tight_layout

    plt.savefig("figures/" +title+"_patternscatterline.png",dpi=300)
    
    plt.show()
    
def plotpattern(pattern,lon,lat):
    plt.figure(figsize=(8,3))

    a1=plt.subplot(1,1,1,projection=ccrs.EqualEarth(central_longitude=180))
    a1.pcolormesh(lon,lat,pattern,vmin=-0.8,vmax=0.8,cmap=cmr.fusion_r,transform=ccrs.PlateCarree())
    c1=a1.contourf(lon,lat,pattern,np.arange(-0.8,0.85,0.05),cmap=cmr.fusion_r,transform=ccrs.PlateCarree())
    a1.coastlines()
    cbar=plt.colorbar(c1)
    cbar.ax.set_ylabel(r'SST ($\sigma$)')
    
    plt.tight_layout()
    plt.show()

def plotpattern_SST(pattern,lon,lat,outputstd):

    lbound = -0.3
    ubound = 0.3
    
    cmapdiff = cmr.fusion_r
    boundsdiff = np.arange(lbound,ubound+0.02,0.02)
    normdiff = colors.BoundaryNorm(boundaries=boundsdiff, ncolors=cmapdiff.N)
    
    pattern = np.squeeze(pattern*outputstd)
    
    plt.figure(figsize=(8,3))

    a1=plt.subplot(1,1,1,projection=ccrs.EqualEarth(central_longitude=180))
    a1.pcolormesh(lon,lat,pattern,norm=normdiff,cmap=cmapdiff,transform=ccrs.PlateCarree())
    c1=a1.contourf(lon,lat,pattern,boundsdiff,cmap=cmr.fusion_r,transform=ccrs.PlateCarree())
    a1.coastlines()
    cbar=plt.colorbar(c1)
    cbar.ax.set_ylabel(r'SST ($^{\circ}$C)')
    cbar.ax.set_yticks(np.arange(lbound,ubound+0.1,0.1))
    
    plt.tight_layout()

    plt.savefig("figures/CMIP6pattern.png",dpi=300)
    
    plt.show()
    
def bestpatternplot_SST(bestpattern,truedata,preddata,outputval,y_pred_val,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SCstd_true,SCstd_pred = allthelinalg.standardizations(outputval,y_pred_val,bestpattern,landmask)
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)#/SCstd_true
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)#/SCstd_pred
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    plt.figure(figsize=(18,5))
    
    a1=plt.subplot(1,3,1,projection=projection)
    c1=a1.pcolormesh(lon,lat,bestpatternplot,vmin=-1,vmax=1,cmap="cmr.fusion_r",transform=transform)
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    plt.title('SC pattern')
    
    plt.subplot(1,3,2)
    plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap='magma')
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:teal')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true SC')
    plt.ylabel('pred SC')
    plt.xlim(-3,3)
    plt.ylim(-3,3)

    plt.subplot(1,3,3)
    plt.plot(yearvec,SC_index_true,color='xkcd:teal',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:golden rod',label='pred')
    plt.ylim(-3,3)
    plt.xlim(1865,2022)
    plt.legend()
    plt.ylabel('SC index')
    plt.xlabel('year')


    plt.suptitle(title,fontsize=30)

    plt.tight_layout()
    plt.show()    

def prettyscatterplot(modeldata,obsval,modellist,testvariants,ylabel,obslabels,savestr):
    
    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))

    colorlist = ["xkcd:teal","xkcd:golden rod"]
    
    lowerbound = np.min(modeldata,axis=1)
    upperbound = np.max(modeldata,axis=1)
    
    lowers = np.mean(modeldata,axis=1)-lowerbound
    uppers = upperbound-np.mean(modeldata,axis=1)
    
    errs = np.concatenate((lowers[np.newaxis,:],uppers[np.newaxis,:]),axis=0)
    
    plt.figure(figsize=(10,4))

    plt.errorbar(np.arange(nmodels),np.mean(modeldata,axis=1),errs,ls='none',color='xkcd:slate')
    for i in range(nmodels):
        xvec = i*np.ones(len(testvariants))
        plt.scatter(xvec,modeldata[i,:])
    for iline in range(len(obsval)):
        plt.hlines(obsval[iline],0,nmodels-1,color=colorlist[iline],label=obslabels[iline])
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.ylim(-0.2,1)
    
    plt.tight_layout()

    plt.savefig(savestr,dpi=300)
    plt.show()

def prettyscatterplot_multiobs(modeldata,obsval,modellist,testvariants,ylabel,source,savestr):

    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))
    
    lowerbound = np.min(modeldata,axis=1)
    upperbound = np.max(modeldata,axis=1)
    
    lowers = np.mean(modeldata,axis=1)-lowerbound
    uppers = upperbound-np.mean(modeldata,axis=1)
    
    errs = np.concatenate((lowers[np.newaxis,:],uppers[np.newaxis,:]),axis=0)

    colors = ["xkcd:dark teal",
             "xkcd:dark gold"]
    
    plt.figure(figsize=(10,4))

    plt.errorbar(np.arange(nmodels),np.mean(modeldata,axis=1),errs,ls='none',color='xkcd:slate',zorder=0)
    for i in range(nmodels):
        xvec = i*np.ones(len(testvariants))
        plt.scatter(xvec,modeldata[i,:],zorder=i+1)
    for iobs,obs in enumerate(obsval):
        plt.scatter(np.arange(nmodels),obs,marker='x',color=colors[iobs],zorder=nmodels+1,label=source[iobs])

    plt.legend()
    
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.ylim(-0.2,1)
    
    plt.tight_layout()

    plt.savefig(savestr,dpi=300)
    plt.show()

def inputplots(inputdata,outputdata,bestpattern,landmask,inres,titlestr):
    
    projection = ccrs.EqualEarth(central_longitude=180)
    transform = ccrs.PlateCarree()

    lonplot = np.arange(0+inres/2,360+inres/2,inres)
    latplot = np.arange(-90+inres/2,90+inres/2,inres)

    SCtimeseries = allthelinalg.index_timeseries(outputdata,bestpattern,landmask)
    
    plt.figure(figsize=(15,5))

    for iplot in range(2):

        title = titlestr + r" $\tau$=" + str(5*(iplot-2)) + " to " + str(5*(iplot-1)) + " years"
        
        plotdata = np.mean(SCtimeseries[:,np.newaxis,np.newaxis]*inputdata[:,:,:,iplot],axis=0)

        a1=plt.subplot(1,2,iplot+1,projection=projection)
        c1=a1.contourf(lonplot,latplot,plotdata,np.arange(-0.5,0.55,0.05),transform=transform,cmap=cmr.fusion_r,extend='both')
        a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor="xkcd:slate"))
        plt.title(title)

    cax = plt.axes((0.92,0.2,0.015,0.6))
    cbar = plt.colorbar(c1,cax=cax)
    cbar.ax.set_ylabel(r'SST anomaly ($^{\circ}$C)')

    #plt.tight_layout()
    plt.show()

