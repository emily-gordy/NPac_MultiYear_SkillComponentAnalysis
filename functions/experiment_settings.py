#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:19:28 2024

@author: egordon4
"""

import numpy as np

def get_experiment_settings(experimentname):

    experiment_dict={
        
            "allcmodel-tos_allcmodel-tos_1-5yearlead": {
            "filename": "allcmodel-tos_allcmodel-tos_1-5yearlead",
            "leadtime": 0,
            "run": 5,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 50,
            "patience": 10,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        }, 
        
        "allcmodel-tos_allcmodel-tos_1-5yearlead_leaveoneout": {
            "filename": "allcmodel-tos_allcmodel-tos_1-5yearlead_leaveoneout",
            "leadtime": 0,
            "run": 5,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 50,
            "patience": 10,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        },
        
        "allcmodel-tos_allcmodel-tos_1-5yearlead_tvtfolds": {
            "filename": "allcmodel-tos_allcmodel-tos_1-5yearlead_tvtfolds",
            "leadtime": 0,
            "run": 5,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 50,
            "patience": 10,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "ntrainvariants": 15,
            "nvalvariants": 8,
            "ntestvariants": 7,
            "foldseeds": [457,841,389,544,365,648,248,149,139,987,453,751,864,745,314]
    
        }, 
        
        "allcmodel-tos_allcmodel-tos_1-3yearlead": {
            "filename": "allcmodel-tos_allcmodel-tos_1-3yearlead",
            "leadtime": 0,
            "run": 3,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 50,
            "patience": 10,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        }, 
        
        "allcmodel-tos_allcmodel-tos_2-9yearlead": {
            "filename": "allcmodel-tos_allcmodel-tos_2-9yearlead",
            "leadtime": 1,
            "run": 8,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [3,3,3],
            # "kernel_size": [5,3,3],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.0001,            
            "learning_rate": 0.1,
            "batch_size": 512,
            "n_epochs": 100,
            "patience": 10,
            "lr_patience":5,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        }, 

        "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg": {
            "filename": "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg",
            "leadtime": 1,
            "inputrun": 5,
            "outputrun": 8,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [3,3,3],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.00001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 100,
            "patience": 10,
            "lr_patience":5,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        }, 

        "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg_leaveoneout": {
            "filename": "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg_leaveoneout",
            "leadtime": 1,
            "inputrun": 5,
            "outputrun": 8,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.0001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 100,
            "patience": 10,
            "lr_patience":5,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        }, 
        
        "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg_tvtfolds": {
            "filename": "allcmodel-tos_allcmodel-tos_2-9yearlead_flexavg_tvtfolds",
            "leadtime": 1,
            "inputrun": 5,
            "outputrun": 8,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[100],
            "ridgepen": 0.0001,            
            "learning_rate": 0.5,
            "batch_size": 512,
            "n_epochs": 100,
            "patience": 10,
            "lr_patience":5,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    7494135,
                    8245643,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "ntrainvariants": 15,
            "nvalvariants": 8,
            "ntestvariants": 7,
            "foldseeds": [457,841,389,544,365,648,248,149,139,987,453,751,864,745,314]
    
        }, 

        "allcmodel-tos_allcmodel-tos_1-5yearlead_keepone": {
            "filename": "allcmodel-tos_allcmodel-tos_1-5yearlead_keepone",
            "leadtime": 0,
            "run": 5,
            "varin": "tos",
            "varout": "tos",
            "filters": [8,8,8],
            "kernel_size": [5,5,5],
            "maxpools": [2,2,2],
            "hiddens":[10],
            "ridgepen": 0.001,            
            "learning_rate": 0.1,
            "batch_size": 32,
            "n_epochs": 50,
            "patience": 10,
            "dropout_rate": 0.2,
            "seeds": [
                    4487873,
                    5487191,
                    5974863,
                    ],
            "activation":"relu",
            "modellist":[
                "ACCESS-ESM1-5",
                "CanESM5",
                "CESM2",
                "CNRM-CM6-1",
                "IPSL-CM6A-LR",
                "MIROC-ES2L",
                "MIROC6",
                "MPI-ESM1-2-LR",
                "NorCPM1",
                ],
            "inbounds": [],
            "outbounds": [20,60,110,260],
            "inres": 4,
            "outres": 4,
            "trainvariants": np.arange(15),
            "valvariants": np.arange(15,23),
            "testvariants": np.arange(23,30),
    
        },
        
    }
    
    return experiment_dict[experimentname]
