# Daan Kivits, 2023

# This file contains functions used by the FLEXPART flux file creation scripts.

##############################################
########## LOAD NECCESSARY PACKAGES ##########
##############################################
from dateutil.relativedelta import relativedelta
import shutil
from datetime import datetime, timedelta, time, date
import os
import xarray as xr
import glob
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import stringtochar
import numpy as np
from scipy import stats

def median2d(arr, new_shape):
    """ Function to average any given shape, which is a multiple of the domain size, to the domain size
    Input:
        arr: np.ndarray: original array to be averaged
        new_shape: tuple: shape to be averaged to
    Returns:
        np.ndarray: averaged arr"""
    shape = (new_shape[0], arr.shape[-2] // new_shape[-2],
             new_shape[1], arr.shape[-1] // new_shape[-1])
    if len(arr.shape) == 3: # If time is included:
        shape = (len(arr),) + shape
        
    a = stats.mode(arr.reshape(shape), axis=1)[0]
    b = stats.mode(a, axis=3)[0]
    return b.squeeze()
def get_lu(flux_array):
    """ Function to extract the landuse given any given shape. The shape should be a multiplication of 
    0.05 x 0.05 degrees, so a shape with a 0.1 x 0.2 gridcell size would be possible, but a 0.0825 x 0.125 wouldn't be.   
    Returns:
        returns the landuse array from the landuse dataset  of any given shape overlapping with the 
        extent of this landuse dataset.

    The following SiB4 landuse classes correspond to these class names:
    1: Desert or bare ground
    2: Evergreen needleleaf forest
    4: Deciduous needleleaf forest
    5: Evergreen broadfleaf forest
    8: Deciduous broadleaf forest
    11: Shrublands (non-tundra)
    12: Shrublands (tundra)
    13: C3 plants
    14: C3 grass
    15: C4 grass
    17: C3 crops
    18: C4 crops
    20: Maize
    22: Soybean
    24: Winter wheat
    
    """
    with nc.Dataset('/projects/0/ctdas/NRT/data/SiB/CORINE_PFT_EUROPA_NRT.nc') as ds:
        lu = ds['landuse'][:]
        lu = np.flipud(lu)
    lu = median2d(lu, flux_array.shape[1:])  
    return lu

def exchange_fluxes_based_on_landuse(src_file, trg_file, lu_type, variablelist):
    """ Function to extract fluxes of two different years, and fill in fluxes of one year 
    to the flux field of the other year for all pixels of a certain landuse type. Plot the difference between
    the input and output flux fields and show the image for each of the variables in variablelist. 

    The following SiB4 landuse classes correspond to these class names:
    1: Desert or bare ground
    2: Evergreen needleleaf forest
    4: Deciduous needleleaf forest
    5: Evergreen broadfleaf forest
    8: Deciduous broadleaf forest
    11: Shrublands (non-tundra)
    12: Shrublands (tundra)
    13: C3 plants
    14: C3 grass
    15: C4 grass
    17: C3 crops
    18: C4 crops
    20: Maize
    22: Soybean
    24: Winter wheat

    """

    with nc.Dataset('/projects/0/ctdas/NRT/data/SiB/CORINE_PFT_EUROPA_NRT.nc') as ds:
        lu = ds['landuse'][:]
        lu = np.flipud(lu)
    
    src = nc.Dataset(src_file, mode = 'r')
    
    ## Create a time variable to loop over
    timevar = src.variables['time'][:]
    fluxname = list(src.variables.keys())[3]

    for timeindex,time in enumerate(timevar):
        trg = nc.Dataset(trg_file, mode='r+')
                
        # Loop over variables of interest and put in lu_specific fluxsets of different year
        if fluxname in variablelist:
            var = trg.variables[fluxname][timeindex,:,:]
            luvar = src.variables[fluxname][timeindex,:,:]

            lu = median2d(lu, luvar.shape[1:])  
            
            var[lu == lu_type] = luvar[lu == lu_type]
            trg.variables[fluxname][timeindex,:,:] = var
            dif = trg.variables[fluxname][timeindex,:,:] - src.variables[fluxname][timeindex,:,:]
            
            if timeindex == 2:
                plt.imshow(np.flipud(dif))
                plt.show()

        # Save the file
        trg.close()
    src.close()

def fill_landuse_with_zeroes(src_file, trg_file, lu_type, variablelist):
    """ Function used to fill a given landuse type (based on the SiB4 PFTs) with zeroes.

    The following SiB4 landuse classes correspond to these class names:
    1: Desert or bare ground
    2: Evergreen needleleaf forest
    4: Deciduous needleleaf forest
    5: Evergreen broadfleaf forest
    8: Deciduous broadleaf forest
    11: Shrublands (non-tundra)
    12: Shrublands (tundra)
    13: C3 plants
    14: C3 grass
    15: C4 grass
    17: C3 crops
    18: C4 crops
    20: Maize
    22: Soybean
    24: Winter wheat

    """

    with nc.Dataset('/projects/0/ctdas/NRT/data/SiB/CORINE_PFT_EUROPA_NRT.nc') as ds:
        lu = ds['landuse'][:]
        lu = np.flipud(lu)
    
    src = nc.Dataset(src_file, mode = 'r')
    
    ## Create a time variable to loop over
    timevar = src.variables['time'][:]
    fluxname = list(src.variables.keys())[3]

    for timeindex,time in enumerate(timevar):
        trg = nc.Dataset(trg_file, mode='r+')
                
        # Loop over variables of interest and put in lu_specific fluxsets of different year
        if fluxname in variablelist:
            var = trg.variables[fluxname][timeindex,:,:]
            luvar = src.variables[fluxname][timeindex,:,:]

            lu = median2d(lu, luvar.shape[1:])  
            
            var[lu == lu_type] = luvar[lu == lu_type]
            trg.variables[fluxname][timeindex,:,:] = var
            dif = trg.variables[fluxname][timeindex,:,:] - src.variables[fluxname][timeindex,:,:]
            
            if timeindex == 2:
                plt.imshow(np.flipud(dif))
                plt.show()

        # Save the file
        trg.close()
    src.close()


def check_mask(src_file, lu_type, variablelist):
    """ Function to check what area is affected by the landuse mask, by simply putting in a large
    emission flux value of 999 and showing the result. """
    
    with nc.Dataset('/projects/0/ctdas/NRT/data/SiB/CORINE_PFT_EUROPA_NRT.nc') as ds:
        lu = ds['landuse'][:]
        lu = np.flipud(lu)

    src = nc.Dataset(src_file)
    
    ## Create a time variable to loop over
    timevar = np.arange(0, len(src.variables['time'][:]),1)
    time_array = np.arange(0, len(timevar),1)
    
    for name, var in src.variables.items():
        if name in variablelist:
            for time in time_array:
                data = src.variables[name][time,:,:]

                lu = median2d(data, data.shape[1:])

                data[lu == lu_type] = 999
                
                plt.imshow(np.flipud(data))
                plt.show()

def create_fluxfile_from_source(src_file, trg_file):
    """ Function to copy the variables and (global and variable) attributes of an existing emission flux file
    into a new emission flux file. """
    src = nc.Dataset(src_file)
    trg = nc.Dataset(trg_file, mode='w')

    # Create the dimensions of the file
    for name, dim in src.dimensions.items():
        trg.createDimension(name, len(dim) if not dim.isunlimited() else None)

    # Copy the global attributes
    trg.setncatts({a:src.getncattr(a) for a in src.ncattrs()})

    # Create the variables in the file
    for name, var in src.variables.items():
        trg.createVariable(name, var.dtype, var.dimensions)

        # Copy the variable attributes
        trg.variables[name].setncatts({a:var.getncattr(a) for a in var.ncattrs()})

        # Copy the variables values (as 'f4' eventually)
        trg.variables[name][:] = src.variables[name][:]

    # Save the file
    trg.close()

def add_variables(dset, listofvars):
        data = 0
        for var in listofvars:
            data = data + dset[var][:,:,:] # slice of each variable
        return data 