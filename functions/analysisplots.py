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

def patternplots_SST(bestpattern,PDOpattern,truedata,preddata,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)
    
    PDOindex_true = allthelinalg.index_timeseries(truedata,PDOpattern,landmask)
    PDOindex_pred = allthelinalg.index_timeseries(preddata,PDOpattern,landmask)
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    PDOpatternplot = np.mean(truedata*PDOindex_true[:,np.newaxis,np.newaxis],axis=0)
    
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    PDOpatternplot = PDOpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    cc_PDO,_ = pearsonr(PDOindex_true,PDOindex_pred)
    
    plt.figure(figsize=(19,8))
    
    a1=plt.subplot(2,3,1,projection=projection)
    a1.pcolormesh(lon,lat,bestpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar1.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' first skill component pattern')
    plt.text(0.02,1.18,"a.",transform=a1.transAxes,fontsize=18,weight='bold')
    
    a2=plt.subplot(2,3,2)
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.text(-2.5,2,"r = %.2f" %(cc_SC))
    plt.xlabel('true skill component index')
    plt.ylabel('pred skill component index')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')
    plt.text(0.02,0.92,"b.",transform=a2.transAxes,fontsize=18,weight='bold')
    
    a3=plt.subplot(2,3,3)
    plt.plot(yearvec,SC_index_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:teal',label='pred')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('skill component index')
    plt.xlabel('year')
    plt.text(0.02,0.92,"c.",transform=a3.transAxes,fontsize=18,weight='bold')

    a4=plt.subplot(2,3,4,projection=projection)
    a4.pcolormesh(lon,lat,PDOpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c4=a4.contourf(lon,lat,PDOpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a4.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar4=plt.colorbar(c4,location='bottom')
    cbar4.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar4.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' PDO pattern')
    plt.text(0.02,1.18,"d.",transform=a4.transAxes,fontsize=18,weight='bold')
    
    a5=plt.subplot(2,3,5)
    c=plt.scatter(PDOindex_true,PDOindex_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.xlabel('true PDO index')
    plt.ylabel('pred PDO index')
    plt.text(-2.5,2.1,"r = %.2f" %(cc_PDO))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat2=plt.colorbar(c)
    cbarscat2.ax.set_ylabel('year')
    plt.text(0.02,0.92,"e.",transform=a5.transAxes,fontsize=18,weight='bold')    
    
    a6=plt.subplot(2,3,6)
    plt.plot(yearvec,PDOindex_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,PDOindex_pred,color='xkcd:teal',label='pred')
    # plt.legend()
    plt.ylabel('PDO index')
    plt.xlabel('year')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.text(0.02,0.92,"f.",transform=a6.transAxes,fontsize=18,weight='bold')    
    # plt.suptitle(title,fontsize=30)

    plt.tight_layout

    plt.savefig("figures/" +title+"_patternscatterline.png",dpi=300)
    
    plt.show()


def patternplots_better_SST(bestpattern,PDOpattern,truedata,preddata,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)
    
    PDOindex_true = allthelinalg.index_timeseries(truedata,PDOpattern,landmask)
    PDOindex_pred = allthelinalg.index_timeseries(preddata,PDOpattern,landmask)
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    PDOpatternplot = np.mean(truedata*PDOindex_true[:,np.newaxis,np.newaxis],axis=0)
    
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    PDOpatternplot = PDOpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    cc_PDO,_ = pearsonr(PDOindex_true,PDOindex_pred)
    
    plt.figure(figsize=(13,8))
    
    a1=plt.subplot(2,2,1,projection=projection)
    a1.pcolormesh(lon,lat,bestpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar1.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' first skill component pattern')
    plt.text(0.02,1.18,"a.",transform=a1.transAxes,fontsize=18,weight='bold')
    
    # a2=plt.subplot(2,3,2)
    # c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    # plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')

    # plt.xlabel('true skill component index')
    # plt.ylabel('pred skill component index')
    # plt.xlim(-3,3)
    # plt.ylim(-3,3)
    # cbarscat1=plt.colorbar(c)
    # cbarscat1.ax.set_ylabel('year')
    # plt.text(0.02,0.92,"b.",transform=a2.transAxes,fontsize=18,weight='bold')
    
    a3=plt.subplot(2,2,2)
    plt.plot(yearvec,SC_index_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:teal',label='pred')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('skill component index')
    plt.xlabel('year')
    plt.text(1990,2.5,"r = %.2f" %(cc_SC),fontsize=14)
    plt.text(0.02,0.92,"b.",transform=a3.transAxes,fontsize=18,weight='bold')

    a4=plt.subplot(2,2,3,projection=projection)
    a4.pcolormesh(lon,lat,PDOpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c4=a4.contourf(lon,lat,PDOpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a4.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar4=plt.colorbar(c4,location='bottom')
    cbar4.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar4.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' PDO pattern')
    plt.text(0.02,1.18,"c.",transform=a4.transAxes,fontsize=18,weight='bold')
    
    # a5=plt.subplot(2,3,5)
    # c=plt.scatter(PDOindex_true,PDOindex_pred,c=yearvec,cmap=cmr.ember)
    # plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    # plt.xlabel('true PDO index')
    # plt.ylabel('pred PDO index')

    # plt.xlim(-3,3)
    # plt.ylim(-3,3)
    # cbarscat2=plt.colorbar(c)
    # cbarscat2.ax.set_ylabel('year')  
    
    a6=plt.subplot(2,2,4)
    plt.plot(yearvec,PDOindex_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,PDOindex_pred,color='xkcd:teal',label='pred')
    # plt.legend()
    plt.ylabel('PDO index')
    plt.xlabel('year')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.text(1990,2.5,"r = %.2f" %(cc_PDO),fontsize=14)
    plt.text(0.02,0.92,"d.",transform=a6.transAxes,fontsize=18,weight='bold')    
    # plt.suptitle(title,fontsize=30)

    plt.tight_layout

    plt.savefig("figures/" +title+"_patternline.png",dpi=300)
    
    plt.show()

def patternplots_SST_predrange(bestpattern,PDOpattern,truedata,preddata,predrange,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)
    
    PDOindex_true = allthelinalg.index_timeseries(truedata,PDOpattern,landmask)
    PDOindex_pred = allthelinalg.index_timeseries(preddata,PDOpattern,landmask)
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    PDOpatternplot = np.mean(truedata*PDOindex_true[:,np.newaxis,np.newaxis],axis=0)
    
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    PDOpatternplot = PDOpatternplot*np.squeeze(outputstd)

    if len(predrange.shape)==3:    
        maxpred = np.max(predrange,axis=(0,1))
        minpred = np.min(predrange,axis=(0,1))

    elif len(predrange.shape)==2:
        maxpred = np.max(predrange,axis=0)
        minpred = np.min(predrange,axis=0)

    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    cc_PDO,_ = pearsonr(PDOindex_true,PDOindex_pred)
    
    plt.figure(figsize=(19,8))
    
    a1=plt.subplot(2,3,1,projection=projection)
    a1.pcolormesh(lon,lat,bestpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar1.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' first skill component pattern')
    plt.text(0.02,1.18,"a.",transform=a1.transAxes,fontsize=18,weight='bold')
    
    a2=plt.subplot(2,3,2)
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true skill component index')
    plt.ylabel('pred skill component index')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')
    plt.text(0.02,0.92,"b.",transform=a2.transAxes,fontsize=18,weight='bold')
    
    a3=plt.subplot(2,3,3)
    plt.fill_between(yearvec,minpred,maxpred,color="xkcd:teal",alpha=0.2)
    plt.plot(yearvec,SC_index_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:teal',label='pred')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('skill component index')
    plt.xlabel('year')
    plt.text(0.02,0.92,"c.",transform=a3.transAxes,fontsize=18,weight='bold')

    a4=plt.subplot(2,3,4,projection=projection)
    a4.pcolormesh(lon,lat,PDOpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c4=a4.contourf(lon,lat,PDOpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a4.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar4=plt.colorbar(c4,location='bottom')
    cbar4.ax.set_xlabel(r'SST ($^{\circ}$C)')
    cbar4.ax.set_xticks(np.arange(-0.6,0.8,0.2))
    plt.title(title + ' PDO pattern')
    plt.text(0.02,1.18,"d.",transform=a4.transAxes,fontsize=18,weight='bold')
    
    a5=plt.subplot(2,3,5)
    c=plt.scatter(PDOindex_true,PDOindex_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.xlabel('true PDO index')
    plt.ylabel('pred PDO index')
    plt.text(-2.5,2.1,"r = %.4f" %(cc_PDO))
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat2=plt.colorbar(c)
    cbarscat2.ax.set_ylabel('year')
    plt.text(0.02,0.92,"e.",transform=a5.transAxes,fontsize=18,weight='bold')    
    
    a6=plt.subplot(2,3,6)
    plt.plot(yearvec,PDOindex_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,PDOindex_pred,color='xkcd:teal',label='pred')
    # plt.legend()
    plt.ylabel('PDO index')
    plt.xlabel('year')
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.text(0.02,0.92,"f.",transform=a6.transAxes,fontsize=18,weight='bold')    
    # plt.suptitle(title,fontsize=30)

    plt.tight_layout

    # plt.savefig("figures/" +title+"_patternscatterline.png",dpi=300)
    
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

def plotpattern_SST(pattern,lon,lat,outputstd,savestr):

    # lbound = -0.3
    # ubound = 0.3

    lbound = -0.6
    ubound = 0.6
    
    cmapdiff = cmr.fusion_r
    boundsdiff = np.arange(lbound,ubound+0.02,0.02)
    # boundsdiff = np.arange(lbound,ubound+0.05,0.05)
    normdiff = colors.BoundaryNorm(boundaries=boundsdiff, ncolors=cmapdiff.N)
    
    pattern = np.squeeze(pattern*outputstd)
    
    plt.figure(figsize=(8,3))

    a1=plt.subplot(1,1,1,projection=ccrs.EqualEarth(central_longitude=180))
    a1.pcolormesh(lon,lat,pattern,norm=normdiff,cmap=cmapdiff,transform=ccrs.PlateCarree())
    c1=a1.contourf(lon,lat,pattern,boundsdiff,cmap=cmr.fusion_r,transform=ccrs.PlateCarree(),extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))
    # a1.coastlines()
    cbar=plt.colorbar(c1)
    cbar.ax.set_ylabel(r'SST ($^{\circ}$C)')
    cbar.ax.set_yticks(np.arange(lbound,ubound+0.1,0.1))
    # cbar.ax.set_yticks(np.arange(lbound,ubound+0.2,0.2))   
    plt.tight_layout()
    if savestr:
        plt.savefig(savestr,dpi=300)
    
    plt.show()
    
def bestpatternplot_SST(bestpattern,truedata,preddata,outputval,y_pred_val,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    plt.figure(figsize=(18,5))
    
    a1=plt.subplot(1,3,1,projection=projection)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    plt.title('SC pattern')
    
    plt.subplot(1,3,2)
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true SC')
    plt.ylabel('pred SC')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')

    plt.subplot(1,3,3)
    plt.plot(yearvec,SC_index_true,color='xkcd:slate grey',label='true')
    plt.plot(yearvec,SC_index_pred,color='xkcd:teal',label='pred')
    plt.ylim(-3,3)
    plt.xlim(1865,2022)
    plt.legend()
    plt.ylabel('SC index')
    plt.xlabel('year')

    plt.suptitle(title,fontsize=30)

    plt.tight_layout()

    plt.savefig("figures/" +title+"_patternscatterline_SConly.png",dpi=800)
    
    plt.show()    

def prettyscatterplot(modeldata,obsval,modellist,testvariants,ylabel,obslabels,savestr):
    
    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))

    colorlist = ["xkcd:teal","xkcd:maroon"]
    
    lowerbound = np.min(modeldata,axis=1)
    upperbound = np.max(modeldata,axis=1)
    
    lowers = np.mean(modeldata,axis=1)-lowerbound
    uppers = upperbound-np.mean(modeldata,axis=1)
    
    errs = np.concatenate((lowers[np.newaxis,:],uppers[np.newaxis,:]),axis=0)
    
    plt.figure(figsize=(8,4))

    plt.errorbar(np.arange(nmodels),np.mean(modeldata,axis=1),errs,ls='none',color='xkcd:slate')
    for i in range(nmodels):
        xvec = i*np.ones(len(testvariants))
        plt.scatter(xvec,modeldata[i,:])
    for iline in range(len(obsval)):
        plt.hlines(obsval[iline],-0.4,nmodels-0.6,color=colorlist[iline],label=obslabels[iline])
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.ylim(-0.2,1)
    plt.xlim(-0.5,nmodels-0.5)
    
    plt.tight_layout()
    if savestr:
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
             "xkcd:maroon"]

    markers = ["x","+"]
    
    plt.figure(figsize=(8,4))

    plt.errorbar(np.arange(nmodels),np.mean(modeldata,axis=1),errs,ls='none',color='xkcd:slate',zorder=0)
    for i in range(nmodels):
        xvec = i*np.ones(len(testvariants))
        plt.scatter(xvec,modeldata[i,:],zorder=i+1,alpha=0.7,marker='o')
    for iobs,obs in enumerate(obsval):
        plt.scatter(np.arange(nmodels),obs,marker=markers[iobs],color=colors[iobs],zorder=nmodels+1,label=source[iobs])

    plt.legend(loc='lower right')
    plt.xlim(-0.6,8.6)
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.ylim(-0.2,1)
    
    plt.tight_layout()
    if savestr:
        plt.savefig(savestr,dpi=300)
    plt.show()

def prettyviolinplot_multiobs(modeldata,obsval,modellist,ylabel,source,savestr):

    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))
    
    colors = ["xkcd:dark teal",
             "xkcd:maroon"]

    markers = ["x","+"]
    
    plt.figure(figsize=(8,4))
    for imodel in range(len(modellist)):
        plt.violinplot(modeldata[imodel,:],positions=[imodel])
    for iobs,obs in enumerate(obsval):
        plt.scatter(np.arange(nmodels),obs,marker=markers[iobs],color=colors[iobs],zorder=nmodels+1,label=source[iobs])

    # plt.legend(loc='upper right')
    
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.ylim(-0.2,1)

    plt.xlim(-0.6,8.6)
    
    plt.tight_layout()
    if savestr:
        plt.savefig(savestr,dpi=300)
    plt.show()

def prettyviolinplot_multiobs_ranks(modeldata,obsval,modellist,ylabel,source,savestr):

    nmodels = len(modellist)
    corrconcat = np.concatenate((modeldata,np.transpose(obsval)),axis=1)
    corrconcat.shape

    scores = []

    for imodel in range(nmodels):

        mat1 = modeldata[imodel]
        mat2 = obsval[:,imodel]

        print(mat2)

        matfull = np.concatenate((mat1,mat2))

        score1 = percentileofscore(matfull,mat2[0])
        score2 = percentileofscore(matfull,mat2[1])

        scores.append([score1,score2])

    scores = np.asarray(scores)

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))
    
    colors = ["xkcd:dark teal",
             "xkcd:maroon"]

    markers = ["x","+"]
    
    plt.figure(figsize=(8,4))
    for imodel in range(len(modellist)):
        plt.violinplot(modeldata[imodel,:],positions=[imodel],showmedians=True)
    for iobs,obs in enumerate(obsval):
        plt.scatter(np.arange(nmodels),obs,marker=markers[iobs],color=colors[iobs],zorder=nmodels+1,label=source[iobs])
        # for imodel in range(nmodels):
        #     plt.text(imodel-0.25,0.81+iobs/10,str(int(scores[imodel,iobs]))+"th",color=colors[iobs])
    plt.legend(loc='lower left',framealpha = 0.2)
    
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.ylim(-0.2,1)
    
    plt.tight_layout()
    if savestr:
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

def bestpatternplot_poster(bestpattern,truedata,preddata,outputval,y_pred_val,landmask,lon,lat,yearvec,title,outputstd):
    
    centre = np.asarray((lon[0]+lon[-1])/2)
    projection = ccrs.EqualEarth(central_longitude=centre)
    transform = ccrs.PlateCarree()
    
    continents = "gray"
    
    SC_index_true = allthelinalg.index_timeseries(truedata,bestpattern,landmask)#/SCstd_true
    SC_index_pred = allthelinalg.index_timeseries(preddata,bestpattern,landmask)#/SCstd_pred
    
    bestpatternplot = np.mean(truedata*SC_index_true[:,np.newaxis,np.newaxis],axis=0)
    bestpatternplot = bestpatternplot*np.squeeze(outputstd)
    
    cc_SC,_ = pearsonr(SC_index_true,SC_index_pred)
    plt.figure(figsize=(6,5))
    
    a1=plt.subplot(1,1,1,projection=projection)
    a1.pcolormesh(lon,lat,bestpatternplot,vmin=-0.6,vmax=0.6,cmap="cmr.fusion_r",transform=transform)
    c1=a1.contourf(lon,lat,bestpatternplot,np.arange(-0.6,0.65,0.05),cmap="cmr.fusion_r",transform=transform,extend='both')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=continents))
    cbar1=plt.colorbar(c1,location='bottom')
    cbar1.ax.set_xlabel(r'SST ($^{\circ}$C)')
    plt.title('SC pattern')

    plt.tight_layout()
    plt.savefig("figures/" +title+"_patternscatterline_SConly1.png",dpi=300)

    plt.show()    
    
    plt.figure(figsize=(6,5))
    c=plt.scatter(SC_index_true,SC_index_pred,c=yearvec,cmap=cmr.ember)
    plt.plot(np.arange(-2,3),np.arange(-2,3),color='xkcd:slate grey')
    plt.text(-2.5,2,"r = %.4f" %(cc_SC))
    plt.xlabel('true SC')
    plt.ylabel('pred SC')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    cbarscat1=plt.colorbar(c)
    cbarscat1.ax.set_ylabel('year')

    plt.tight_layout()
    plt.savefig("figures/" +title+"_patternscatterline_SConly2.png",dpi=300)

    plt.show()    

    plt.figure(figsize=(8,5))
    plt.plot(yearvec,SC_index_true,color='xkcd:slate grey',label='true',linewidth=1.8)
    plt.plot(yearvec,SC_index_pred,color='xkcd:teal',label='pred',linewidth=2.2)
    plt.ylim(-3,3)
    plt.xlim(yearvec[0],yearvec[-1])
    plt.legend()
    plt.ylabel('SC index')
    plt.xlabel('year')

    plt.tight_layout()
    plt.savefig("figures/" +title+"_patternscatterline_SConly3.png",dpi=300)
    
    plt.show()    

def prettyscatterplot_supp(modeldata,obsval,modellist,testvariants,ylabel,obslabels,savestr,legend):
    
    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))

    colorlist = ["xkcd:teal","xkcd:maroon"]
    
    lowerbound = np.min(modeldata,axis=1)
    upperbound = np.max(modeldata,axis=1)
    
    lowers = np.mean(modeldata,axis=1)-lowerbound
    uppers = upperbound-np.mean(modeldata,axis=1)
    
    errs = np.concatenate((lowers[np.newaxis,:],uppers[np.newaxis,:]),axis=0)
    
    plt.figure(figsize=(8,4))

    plt.errorbar(np.arange(nmodels),np.mean(modeldata,axis=1),errs,ls='none',color='xkcd:slate')
    for i in range(nmodels):
        xvec = i*np.ones(len(testvariants))
        plt.scatter(xvec,modeldata[i,:])
    for iline in range(len(obsval)):
        plt.hlines(obsval[iline],-0.4,nmodels-0.6,color=colorlist[iline],label=obslabels[iline])
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(loc='lower left')
    
    plt.ylim(-0.5,1)
    plt.xlim(-0.5,nmodels-0.5)
    
    plt.tight_layout()
    if savestr:
        plt.savefig(savestr,dpi=300)
    plt.show()

def varexplained(y_pred,output,pattern,griddeddata,landmask,latvec,lonvec):

    transform = ccrs.PlateCarree()

    nlatbig = len(latvec)
    nlonbig = len(lonvec)

    SCtimeseries = allthelinalg.index_timeseries(output,pattern,landmask)
    SCtimeseries_pred = allthelinalg.index_timeseries(y_pred,pattern,landmask)

    R_out = np.empty((nlatbig,nlonbig))

    for ilat in range(nlatbig):
        for ilon in range(nlonbig):
            R_out[ilat,ilon],_ = pearsonr(SCtimeseries,griddeddata[:,ilat,ilon])  

    R_out[np.isnan(R_out)] = 0

    lbound = 0
    ubound = 0.8

    cmapinc = cmr.ember
    bounds = np.arange(lbound,ubound+0.05,0.05)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmapinc.N)

    plt.figure(figsize=(8,7))

    a1=plt.subplot(1,1,1,projection=ccrs.EqualEarth(central_longitude=255))
    c=a1.pcolormesh(lonvec,latvec,R_out**2,norm=norm,cmap='inferno',transform=transform)
    a1.coastlines(color='gray')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='grey'))

    cax=plt.axes((0.2,0.2,0.6,0.03))
    cbar=plt.colorbar(c,cax=cax,orientation='horizontal')
    cbar.ax.set_xlabel(r'R$^2$')
    plt.tight_layout()

    plt.savefig("figures/varexplained.png")

    plt.show()           

def prettyviolinplot_supp_summary(modeldata,obs1,obs2,modellist,ylabel,savestr):
    
    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))

    nmodels = len(modellist)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,nmodels)))
    
    colors = ["xkcd:dark teal",
             "xkcd:maroon"]

    markers = ["x","+"]
    
    plt.figure(figsize=(8,4))
    for imodel in range(len(modellist)):
        plt.violinplot(modeldata[:,imodel,:].flatten(),positions=[imodel])
    # for iobs,obs in enumerate(obs1):
    #     plt.hlines(obs,0,len(modellist),color=colors[0],linewidth=0.4)
    # for iobs,obs in enumerate(obs2):
    #     plt.hlines(obs,0,len(modellist),color=colors[1],linewidth=0.4)
    minobs1 = np.min(obs1)
    minobs2 = np.min(obs2)
    maxobs1 = np.max(obs1)
    maxobs2 = np.max(obs2)
    plt.fill_between(np.arange(-0.5,len(modellist)+0.5),minobs1,maxobs1,color=colors[0],alpha=0.2,label="ERSST range")
    plt.fill_between(np.arange(-0.5,len(modellist)+0.5),minobs2,maxobs2,color=colors[1],alpha=0.2,label="HadISST range")

    plt.legend(loc='lower left',framealpha=0.2)
    
    plt.xticks(np.arange(nmodels),labels=modellist,rotation=60)
    plt.ylabel(ylabel)
    plt.ylim(-0.25,1)
    plt.yticks(np.arange(-0.2,1.2,0.2))
    plt.xlim(-0.5,len(modellist)-0.5)
    
    plt.tight_layout()
    if savestr:
        plt.savefig(savestr,dpi=300)
    plt.show()