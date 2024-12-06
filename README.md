## Welcome to the paper code for what is hopefully another excellent journal submission

This repository contains the code for the paper "Identifying a pattern of predictable multi-year North Pacific SST variability in historical observations" (working title). 

## Python Environment
One day, I will learn how to make a yml file, but today is not that day so first of all some environment requirements: All code is written in Python 3.9. On top of the basic packages (numpy, matplotlib, scipy), this project uses 
* [tensorflow 2.11](https://www.tensorflow.org/install) for creating the CNNs.
* [xarray](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html) for preprocessing data
* [xesmf](https://xesmf.readthedocs.io/en/stable/installation.html) for regridding
* [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) to make map plots
* [cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install) because I like these colorbars

## Data
Now for the data, I downloaded and preprocessed SST output from the historical experiments of nine GCMs (below), eight of which are available through [ESGF](https://aims2.llnl.gov/search) and the last (CESM2) is accessed through [Climate Data Gateway at NCAR](https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2le.atm.proc.monthly_ave.SST.html). 
* ACCESS-ESM1-5
* CanESM5
* CESM2
* CNRM-CM6-1
* IPSL-CM6A-LR
* MIROC-ES2L
* MIROC6
* MPI-ESM1-2-LR
* NorCPM1

All are bilinearly regridded to a regular 2x2 grid and also merged into a file a containing all ensemble members prior to beginning this analysis, such that each file is a netcdf containing SST data with dims variant x time x lat x lon and naming convention tos_Omon_\[modelname\]_historical_allens_2x2_185001-201412.nc .

## Code
Congratulations you have made it to the running code portion of this repository. To recreate the main figures from the my work, first run ```train_allmodels.py``` and ensure saveflag is set: ```saveflag=True``` for the first run. The saveflag ensures that once preprocessed into input/output numpy arrays, the data is saved in a compressed format to make it easier to call for later runs, or if one decides to do any hyperparameter tuning. Speaking of hyperparameters, any experiment settings, inluding CNN hyperparameters are found in the rather aptly named ```experiment_settings.py``` in the ```functions/``` directory. I would estimate my code is robust to changing roughly nothing so proceed with caution. 

After running ```train_allmodels.py``` the user is welcomed to open the notebook ```main_analysis_notebook.ipynb``` where most of the main paper figures are generated. The exception being the leave-one-out experiments. To generate those figures, first run ```train_leaveoneout.py```, go away and make a cuppa, then whip through the notebook ```evaluate_leaveoneout.ipynb```. The notebook ```miscfigures.ipynb``` currently contains scripts for generating figures I have found useful to use in presentations and also the surface temperature analysis.

## Supplementary Material
The directory ```supplemental/``` contains scripts and notebooks for checks that are both in the supplementary material.

## Extras for the bottom-scrollers:
Please contact me with questions about this project or related ideas! I think this is a neat methodology with tons of applications and am interested in examining how else it can be applied across Earth system components, temporal or spatial scales. If you are here to bully me about using tensorflow, or generally bad coding, I direct you to send any complaints to your nearest waste water treatment plant. 

Love Dr. Em xx


