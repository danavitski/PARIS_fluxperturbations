# Daan Kivits, 2023

# This file contains functions used by the FLEXPART flux file creation scripts.

##############################################
########## LOAD NECCESSARY PACKAGES ##########
##############################################
from dateutil.relativedelta import relativedelta
import shutil
from datetime import datetime, timedelta, time, date
import os
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import stringtochar
import numpy as np
from scipy import stats

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
        extent of this landuse dataset """
    with nc.Dataset('/projects/0/ctdas/NRT/data/SiB/CORINE_PFT_EUROPA_NRT.nc') as ds:
        lu = ds['landuse'][:]
        lu = np.flipud(lu)
    lu = median2d(lu, flux_array.shape[1:])  
    return lu