#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:15:42 2024

@author: egordon4
"""

import xarray as xr
import glob
import numpy as np
import xesmf as xe

from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

emilypath = "/Users/egordon4/Documents/Experiments/NPac_MultiYear_SkillComponentAnalysis/"
relpath = "data/"
path = emilypath+relpath

def pull_data(var,cmodel):
    filelist = glob.glob(path + var + "*" + cmodel + "*2x2*.nc")
    file = filelist[0]
    
    allens = xr.open_dataarray(file,chunks='auto')
    allens = allens.isel(variant=np.arange(30))
    
    allensannual = allens.groupby("time.year").mean(dim="time")
    ensmean = allensannual.mean(dim="variant")
    allensint = allensannual-ensmean
    
    allensint = allensint.squeeze()

    return allensint

def pull_data_monthly(var,cmodel):
    filelist = glob.glob(path + var + "*" + cmodel + "*2x2*.nc")
    file = filelist[0]
    
    allens = xr.open_dataarray(file,chunks='auto')
    allens = allens.isel(variant=np.arange(30))
    
    ensmean = allens.mean(dim="variant")
    allensint = allens-ensmean
    
    allensint = allensint.squeeze()

    return allensint

def pull_data_obs(var,source):
    
    if source == "ERSST":
    
        filelist = glob.glob(path + "*" + source + "*2x2*.nc")
        file = filelist[0]
        
        ds = xr.open_dataset(file)
        da = ds["sst"]
    elif source == "HadISST":
    
        filelist = glob.glob(path + "*" + source + "*2x2*.nc")
        file = filelist[0]
        
        ds = xr.open_dataset(file)
        da = ds["sst"]
    
    polys = da.polyfit(dim="time",deg=2)    
    coordout = da.time
    trend = xr.polyval(coord=coordout, coeffs=polys.polyfit_coefficients)
    
    da_anom = da-trend

    da_anom = da_anom.groupby("time.year").mean() # annual means

    return da_anom

def pull_data_obs_PDOdemean(var,source):
    
    if source == "ERSST":
    
        filelist = glob.glob(path + "*" + source + "*2x2*.nc")
        file = filelist[0]
        
        ds = xr.open_dataset(file)
        da = ds["sst"]
    elif source == "HadISST":
    
        filelist = glob.glob(path + "*" + source + "*2x2*.nc")
        file = filelist[0]
        
        ds = xr.open_dataset(file)
        da = ds["sst"]
    
    da_seasonal = da.groupby("time.month").mean("time")
    da = da.groupby("time.month")-da_seasonal # remove seasonal cycle
    
    da_subsel = da.sel(lat=slice(-60,60))
    weights = np.cos(np.deg2rad(da_subsel.lat))
    da_weighted=da_subsel.weighted(weights).mean(dim=("lat",'lon')) # remove global mean SST
    
    da_anom = da-da_weighted

    return da_anom

def outlonxlat(settings):
    
    outres = settings["outres"]
    outbounds = settings["outbounds"]
    
    allensout = pull_data(settings["varout"],"CESM2")
    outgrid = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90+outres/2,90+outres/2, outres), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(0+outres/2, 360+outres/2, outres), {"units": "degrees_east"}),
    }
    )
    
    regridder = xe.Regridder(allensout, outgrid, "bilinear", periodic=True, ignore_degenerate=True)
    allensout = regridder(allensout,keep_attrs=True) 

    if outbounds[2]<0:
        allensout.coords['lon'] = (allensout.coords['lon'] + 180) % 360 - 180
        allensout = allensout.sortby(allensout.lon)
        outputdata = allensout.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
    else:
        outputdata = allensout.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))

    lon = outputdata.lon
    lat = outputdata.lat    
    
    return lon,lat

def inlonxlat(settings):
    
    inres = settings["inres"]
    inbounds = settings["inbounds"]
    
    allensout = pull_data(settings["varout"],"CESM2")
    outgrid = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90+inres/2,90+inres/2, inres), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(0+inres/2, 360+inres/2, inres), {"units": "degrees_east"}),
    }
    )
    
    regridder = xe.Regridder(allensout, outgrid, "bilinear", periodic=True, ignore_degenerate=True)
    allensout = regridder(allensout,keep_attrs=True) 

    if inbounds[2]<0:
        allensout.coords['lon'] = (allensout.coords['lon'] + 180) % 360 - 180
        allensout = allensout.sortby(allensout.lon)
        outputdata = allensout.sel(lat=slice(inbounds[0],inbounds[1]),lon=slice(inbounds[2],inbounds[3]))
    else:
        outputdata = allensout.sel(lat=slice(inbounds[0],inbounds[1]),lon=slice(inbounds[2],inbounds[3]))

    lon = outputdata.lon
    lat = outputdata.lat    
    
    return lon,lat

def regrid(da,res):
    outgrid = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90+res/2,90+res/2, res), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(0+res/2, 360+res/2, res), {"units": "degrees_east"}),
    }
    )
    regridder = xe.Regridder(da, outgrid, "bilinear", periodic=True, ignore_degenerate=True)
    da_regrid = regridder(da,keep_attrs=True) 
    return da_regrid

def make_inputoutput_obs(settings,source):
    
    inres = settings["inres"]
    outres = settings["outres"]
    leadtime = settings["leadtime"]
    inbounds = settings["inbounds"]
    outbounds = settings["outbounds"]
    run = settings["run"]    
    
    # hard code year range for obs
    year1 = 1870
    year2 = 2022
    
    obsin = pull_data_obs(settings["varin"],source)
    obsout = pull_data_obs(settings["varout"],source)
    
    obsin = obsin.rolling(year=run,center=False).mean()
    obsout = obsout.rolling(year=run,center=False).mean()

    if inres:
        obsin = regrid(obsin,inres)
    if outres:
        obsout = regrid(obsout,outres)

    if len(inbounds)==0:
        inputdata = obsin.sel(year=slice(year1,year2-(leadtime+run)))
    else:
        if inbounds[2]<0:
            obsin.coords['lon'] = (obsin.coords['lon'] + 180) % 360 - 180
            obsin = obsin.sortby(obsin.lon)

        inputdata = obsin.sel(year=slice(year1,year2-(leadtime+run)),lat=slice(inbounds[0],inbounds[1]),lon=slice(inbounds[2],inbounds[3]))
    if outbounds[2]<0:
        obsout.coords['lon'] = (obsout.coords['lon'] + 180) % 360 - 180
        obsout = obsout.sortby(obsout.lon)

    outputdata = obsout.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]),year=slice(year1+leadtime+2*run,year2))
    
    return inputdata,outputdata

def concatobs(inputobs,outputobs,outputstd,run):
    
    inputobs = np.asarray(inputobs)
    outputobs = np.asarray(outputobs)
    
    inputobs[:,np.isnan(np.mean(inputobs,axis=0))] = 0
    inputobs1 = inputobs[:-1*run,:,:]
    inputobs2 = inputobs[run:,:,:]
    inputobs = np.concatenate((inputobs1[:,:,:,np.newaxis],inputobs2[:,:,:,np.newaxis]),axis=3)
    
    outputobs = outputobs/outputstd
    outputobs[np.isnan(outputobs)] = 0

    return inputobs,outputobs

def make_inputoutput_modellist(settings):

    allinputdata = []
    alloutputdata = []
    
    modellist = settings["modellist"]
    inres = settings["inres"]
    outres = settings["outres"]
    leadtime = settings["leadtime"]
    inbounds = settings["inbounds"]
    outbounds = settings["outbounds"]
    run = settings["run"]
    # n_input = settings["n_input"]
    
    startyear = 1851 # start year for LEs
    year1 = startyear+run # first year with data
    year2 = 2014 # last year with data
    
    for cmodel in modellist:
        print(cmodel)
        allensin = pull_data(settings["varin"],cmodel)
        allensout = pull_data(settings["varout"],cmodel)
    
        allensin = allensin.rolling(year=run,center=False).mean() # look BACK running mean
        allensout = allensout.rolling(year=run,center=False).mean()

        if inres:
            allensin = regrid(allensin,inres)
        if outres:
            allensout = regrid(allensout,outres)
                    
        if len(inbounds)==0:
            inputdata = allensin.sel(year=slice(year1,year2-(leadtime+run)))
        else:
            if inbounds[2]<0:
                allensin.coords['lon'] = (allensin.coords['lon'] + 180) % 360 - 180
                allensin = allensin.sortby(allensin.lon)
                inputdata = allensin.sel(year=slice(year1,year2-(leadtime+2*run)),lat=slice(inbounds[0],inbounds[1]),lon=slice(inbounds[2],inbounds[3]))
            else:
                inputdata = allensin.sel(year=slice(year1,year2-(leadtime+2*run)),lat=slice(inbounds[0],inbounds[1]),lon=slice(inbounds[2],inbounds[3]))
    
        if outbounds[2]<0:
            allensout.coords['lon'] = (allensout.coords['lon'] + 180) % 360 - 180
            allensout = allensout.sortby(allensout.lon)
            outputdata = allensout.sel(year=slice(year1+(leadtime+2*run),year2),lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
        else:
            outputdata = allensout.sel(year=slice(year1+(leadtime+2*run),year2),lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
        
        inputdata=inputdata.assign_coords({"cmodel":cmodel,
                                           "variant":np.arange(30)})
        outputdata=outputdata.assign_coords({"cmodel":cmodel,
                                             "variant":np.arange(30)})
    
        allinputdata.append(inputdata)
        alloutputdata.append(outputdata)
    
    allinputdata = xr.concat(allinputdata,dim="cmodel",coords='minimal')
    alloutputdata = xr.concat(alloutputdata,dim="cmodel",coords='minimal')
    
    allinputdata = allinputdata.transpose("cmodel","variant","year","lat","lon")
    alloutputdata = alloutputdata.transpose("cmodel","variant","year","lat","lon")
    
    return allinputdata,alloutputdata

def splitandflatten(allinputdata,alloutputdata,variantsplit,run):
    
    allinputdata = np.asarray(allinputdata)
    alloutputdata = np.asarray(alloutputdata)
    
    print('conversion done')

    ntrain = len(variantsplit[0])
    nval = len(variantsplit[1])
    ntest = len(variantsplit[2])

    input_train = allinputdata[:,variantsplit[0],:,:,:]
    input_val = allinputdata[:,variantsplit[1],:,:,:]
    input_test = allinputdata[:,variantsplit[2],:,:,:]   
    
    input1_train = input_train[:,:,:-1*run,:,:]
    input2_train = input_train[:,:,run:,:,:]
    input1_val = input_val[:,:,:-1*run,:,:]
    input2_val = input_val[:,:,run:,:,:]
    input1_test = input_test[:,:,:-1*run,:,:]
    input2_test = input_test[:,:,run:,:,:]
    
    shape1 = input1_train.shape
    
    inputtrain1_flat = input1_train.reshape((shape1[0]*shape1[2]*ntrain,shape1[3],shape1[4]))
    inputval1_flat = input1_val.reshape((shape1[0]*shape1[2]*nval,shape1[3],shape1[4]))
    inputtest1_flat = input1_test.reshape((shape1[0]*shape1[2]*ntest,shape1[3],shape1[4]))
    
    inputtrain2_flat = input2_train.reshape((shape1[0]*shape1[2]*ntrain,shape1[3],shape1[4]))
    inputval2_flat = input2_val.reshape((shape1[0]*shape1[2]*nval,shape1[3],shape1[4]))
    inputtest2_flat = input2_test.reshape((shape1[0]*shape1[2]*ntest,shape1[3],shape1[4]))
    
    if len(alloutputdata.shape)==5: # predicting a domain
    
        output_train = alloutputdata[:,variantsplit[0],:,:,:]
        output_val = alloutputdata[:,variantsplit[1],:,:,:]
        output_test = alloutputdata[:,variantsplit[2],:,:,:]
        
        shape2 = output_train.shape
        
        outputtrain_flat = output_train.reshape((shape2[0]*shape2[2]*ntrain,shape2[3],shape2[4]))
        outputval_flat = output_val.reshape((shape2[0]*shape2[2]*nval,shape2[3],shape2[4]))
        outputtest_flat = output_test.reshape((shape2[0]*shape2[2]*ntest,shape2[3],shape2[4]))
    
    elif len(alloutputdata.shape)==3: # or predicting a single point

        output_train = alloutputdata[:,variantsplit[0],:]
        output_val = alloutputdata[:,variantsplit[1],:]
        output_test = alloutputdata[:,variantsplit[2],:]
        
        shape1 = input_train.shape
        shape2 = output_train.shape
    
        outputtrain_flat = output_train.reshape((shape2[0]*shape2[2]*ntrain))
        outputval_flat = output_val.reshape((shape2[0]*shape2[2]*nval))
        outputtest_flat = output_test.reshape((shape2[0]*shape2[2]*ntest))
    
    inputtrain_flat = np.concatenate((inputtrain1_flat[:,:,:,np.newaxis],inputtrain2_flat[:,:,:,np.newaxis]),axis=3)
    inputval_flat = np.concatenate((inputval1_flat[:,:,:,np.newaxis],inputval2_flat[:,:,:,np.newaxis]),axis=3)
    inputtest_flat = np.concatenate((inputtest1_flat[:,:,:,np.newaxis],inputtest2_flat[:,:,:,np.newaxis]),axis=3)
       
    
    return inputtrain_flat,inputval_flat,inputtest_flat,outputtrain_flat,outputval_flat,outputtest_flat

def PDO_pattern_allmodels(settings):
    
    # PDO pattern calculated across all models in the validation set
    
    PDObounds = [20,60,110,260]
    valvariants = settings["valvariants"]
    modellist = settings["modellist"]
    outbounds = settings["outbounds"]
    outres = settings["outres"]
    
    allinputdata = []
    allglob = []
    
    for im,cmodel in enumerate(modellist):
        print(cmodel)
        allensin = pull_data_monthly("tos",cmodel) # all data
        allensglob = allensin.sel(time=slice("1851","2014")).isel(variant=valvariants) # global SST
        #NPac only for PDO calc
        allensin = allensin.sel(lat=slice(PDObounds[0],PDObounds[1]),lon=slice(PDObounds[2],PDObounds[3]),time=slice("1851","2014")).isel(variant=valvariants)
        
        if im == 0: # impose standard datetime format on ALL models
            timefix = allensin.time
        
        allensin = allensin.assign_coords({"cmodel":cmodel,
                                           "variant":np.arange(len(valvariants)),
                                           "time":timefix})   
        allensglob = allensglob.assign_coords({"cmodel":cmodel,
                                           "variant":np.arange(len(valvariants)),
                                           "time":timefix})                       
        
        allinputdata.append(allensin)
        allglob.append(allensglob)
    
    # data for PDO calc
    allinputdata = xr.concat(allinputdata,dim="cmodel",coords='minimal')
    allinputdata = allinputdata.transpose("cmodel","variant","time","lat","lon")
    
    # data to project PDO onto
    allglob = xr.concat(allglob,dim="cmodel",coords='minimal')
    allglob = allglob.transpose("cmodel","variant","time","lat","lon")
    
    # North Pacific coords    
    Paclat = allensin.lat
    Paclon = allensin.lon

    # Global SST coords            
    alllat = allensglob.lat
    alllon = allensglob.lon
    
    time = allinputdata.time
    variant = allinputdata.variant
    
    # weight NPac data by sqrt(cos(latitidue))
    Paclonxlat = np.meshgrid(Paclon,Paclat)[1]
    Pacweights = np.sqrt(np.cos(Paclonxlat*np.pi/180))

    PacificSST = np.squeeze(np.asarray(allinputdata))
    print("np conversion done")
    
    NPacshape = PacificSST.shape

    # flatten and weight
    PacificSST_flat = np.reshape(PacificSST,(len(variant)*len(time)*len(modellist),NPacshape[3],NPacshape[4]))
    PacificSSTw = PacificSST_flat*Pacweights[np.newaxis,:,:]
    # and remove land data
    Pacnoland = PacificSSTw[:,~np.isnan(np.mean(PacificSST_flat,axis=0))]
    
    # take covariance, do it lazily bc we only want 1st eigen value
    PacCov = np.cov(np.transpose(Pacnoland))
    PacCov_s = sparse.csc_matrix(PacCov)
    
    # get evec corresponding to highest eigen value (EOF)
    eigval,evec = eigs(PacCov_s,1)
    evec = np.squeeze(np.real(evec))
    
    print("eof done")
    
    if np.sum(evec)>0: # lil correction to get the right sign PDO
        evec = -1*evec
    
    # calculate PDO index, standardize and reshape to model x ensemble member x time
    PDOindex = np.matmul(Pacnoland,evec)
    PDOindex = (PDOindex-np.mean(PDOindex))/np.std(PDOindex)
    PDOindex = np.reshape(PDOindex,(len(modellist),len(valvariants),len(time)))
    
    da_PDO = xr.DataArray(
        data=PDOindex,
        dims=["cmodel","variant","time"],
        coords=dict(            
            cmodel=modellist,
            time=time,
            variant=variant,
        ),
        attrs=dict(
            description="PDOindex",
            units="covariance",
        ),
    )
    
    print("index calculated")
    
    # project PDO index onto global SST
    globEOF = da_PDO * allglob
    globEOF = globEOF.mean(dim=("time","variant","cmodel"))
    
    print("global pattern calculated")
    
    da_EOF = xr.DataArray(
        data=globEOF,
        dims=["lat","lon"],
        coords={"lat":alllat,
                "lon":alllon,
                })
    
    # regrid and cut pattern to fit experiment domain
    
    if outres:
        outgrid = xr.Dataset(
        {
                "lat": (["lat"], np.arange(-90+outres/2,90+outres/2, outres), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(0+outres/2, 360+outres/2, outres), {"units": "degrees_east"}),
        }
        )
        
        regridder = xe.Regridder(da_EOF, outgrid, "bilinear", periodic=False, ignore_degenerate=True)
        da_EOF = regridder(da_EOF,keep_attrs=True) 
    
        print('regridded (if necessary)')
    
    if outbounds:
        da_EOF = da_EOF.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
    
    PDOpattern = np.asarray(da_EOF)

    return PDOpattern

def PDO_pattern_singlemodel(settings):
    
    # PDO index calculated individually for each model in the validation set, 
    # Index timeseries to correspond to the outputs of the neural network
    
    PDObounds = [20,60,110,260]
    valvariants = settings["valvariants"]
    modellist = settings["modellist"]
    outbounds = settings["outbounds"]
    outres = settings["outres"]
    
    PDOpatterns = []
    
    for cmodel in modellist:
        print(cmodel)
        allensin = pull_data_monthly("tos",cmodel)
        allensglob = allensin.isel(variant=valvariants)
        allensin = allensin.sel(lat=slice(PDObounds[0],PDObounds[1]),lon=slice(PDObounds[2],PDObounds[3])).isel(variant=valvariants)
        
        Paclat = allensin.lat
        Paclon = allensin.lon
        alllat = allensglob.lat
        alllon = allensglob.lon
        time = allensin.time
        variant = allensin.variant

        Paclonxlat = np.meshgrid(Paclon,Paclat)[1]
        Pacweights = np.sqrt(np.cos(Paclonxlat*np.pi/180))
    
        PacificSST = np.squeeze(np.asarray(allensin))
        NPacshape = PacificSST.shape

        PacificSST_flat = np.reshape(PacificSST,(len(valvariants)*len(time),NPacshape[2],NPacshape[3]))

        PacificSSTw = PacificSST_flat*Pacweights[np.newaxis,:,:]
    
        Pacnoland = PacificSSTw[:,~np.isnan(PacificSST_flat[0,:,:])]
        PacCov = np.cov(np.transpose(Pacnoland))
    
        PacCov_s = sparse.csc_matrix(PacCov)
    
        eigval,evec = eigs(PacCov_s,1)
        evec = np.squeeze(np.real(evec))
        
        print("eof done")
        
        if np.sum(evec)>0:
            evec = -1*evec
            
        PDOindex = np.matmul(Pacnoland,evec)
        PDOindex = np.reshape(PDOindex,(len(valvariants),len(time)))
        
        PDOindex = (PDOindex-np.mean(PDOindex))/np.std(PDOindex)
        
        da_PDO = xr.DataArray(
            data=PDOindex,
            dims=["variant","time"],
            coords=dict(
                time=time,
                variant=variant,
            ),
            attrs=dict(
                description="PDOindex",
                units="covariance",
            ),
        )
        
        da_PDO = da_PDO.assign_coords({"cmodel":cmodel})
        
        globEOF = da_PDO * allensglob
        globEOF = globEOF.mean(dim=("time","variant"))
        
        da_EOF = xr.DataArray(
            data=globEOF,
            dims=["lat","lon"],
            coords={"lat":alllat,
                    "lon":alllon,
                    "cmodel":cmodel})
    
        if outres:
            outgrid = xr.Dataset(
            {
                "lat": (["lat"], np.arange(-90+outres/2,90+outres/2, outres), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(0+outres/2, 360+outres/2, outres), {"units": "degrees_east"}),
            }
            )
            
            regridder = xe.Regridder(da_EOF, outgrid, "bilinear", periodic=True, ignore_degenerate=True)
            da_EOF = regridder(da_EOF,keep_attrs=True) 
            
        if outbounds:
            da_EOF = da_EOF.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
        
        PDOpatterns.append(da_EOF)

    PDOpatterns = xr.concat(PDOpatterns,dim="cmodel")
    
    PDOpatterns = np.asarray(PDOpatterns.transpose("cmodel","lat","lon"))
    
    return PDOpatterns

def PDOobs(settings,source):
    
    year1 = 1870
    year2 = 2020
    
    bounds = [20,60,110,260]
    
    outres = settings["outres"]
    outbounds = settings["outbounds"]
    run = settings["run"]
    run=run*12
    
    allensin = pull_data_obs_PDOdemean(settings["varout"],source)
    allensin = allensin.sel(time=slice(str(year1),str(year2)))
    allensinPac = allensin.sel(lat=slice(bounds[0],bounds[1]),lon=slice(bounds[2],bounds[3]))
    
    Paclat = allensinPac.lat
    Paclon = allensinPac.lon            
    alllat = allensin.lat
    alllon = allensin.lon

    Paclonxlat = np.meshgrid(Paclon,Paclat)[1]
    Pacweights = np.sqrt(np.cos(Paclonxlat*np.pi/180))

    PacificSST = np.squeeze(np.asarray(allensinPac))
    print("np conversion done")

    PacificSSTw = PacificSST*Pacweights[np.newaxis,:,:]
    
    Pacnoland = PacificSSTw[:,~np.isnan(np.mean(PacificSST,axis=0))]
    PacCov = np.cov(np.transpose(Pacnoland))

    PacCov_s = sparse.csc_matrix(PacCov)
    
    eigval,evec = eigs(PacCov_s,1)
    evec = np.squeeze(np.real(evec))
    
    print("eof done")
    
    if np.sum(evec)>0:
        evec = -1*evec
            
    PDOindex = np.matmul(Pacnoland,evec)
    
    PDOindex = (PDOindex-np.mean(PDOindex))/np.std(PDOindex)
    
    globEOF = PDOindex[:,np.newaxis,np.newaxis] * allensin
    globEOF = globEOF.mean(dim=("time"))
    
    da_EOF = xr.DataArray(
        data=globEOF,
        dims=["lat","lon"],
        coords={"lat":alllat,
                "lon":alllon,
                })

    if outres:
        outgrid = xr.Dataset(
        {
                "lat": (["lat"], np.arange(-90+outres/2,90+outres/2, outres), {"units": "degrees_north"}),
                "lon": (["lon"], np.arange(0+outres/2, 360+outres/2, outres), {"units": "degrees_east"}),
        }
        )
        
        regridder = xe.Regridder(da_EOF, outgrid, "bilinear", periodic=False, ignore_degenerate=True)
        da_EOF = regridder(da_EOF,keep_attrs=True) 
        
    if outbounds:
        da_EOF = da_EOF.sel(lat=slice(outbounds[0],outbounds[1]),lon=slice(outbounds[2],outbounds[3]))
    
    PDOpattern = da_EOF
    
    return PDOpattern









