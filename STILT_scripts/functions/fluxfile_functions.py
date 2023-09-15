# Daan Kivits, 2023

# This file contains functions used by the FLEXPART flux file creation scripts.
# EDIT 15-09-2023: Added functions to use with the STILT model.

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
import pandas as pd
from functions.background_functions import *
import pickle

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

def coordinate_list(lat_ll, lat_ur, lon_ll, lon_ur, lat_step, lon_step):
    """ Function to create a list of lons and lats given a 
    range and stepsize.
    """
    lats = list(np.round(np.arange(lat_ll, lat_ur, lat_step),4))
    lons = list(np.round(np.arange(lon_ll, lon_ur, lon_step),4))
    return lats,lons

def footprint_unique_months(fp_filelist, simulation_len):
    """ Function to extract the unique months of the footprint files."""
    # Define time range 
    timestr_start = fp_filelist[0][-37:-24]
    timestr_end = fp_filelist[-1][-37:-24]
    fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')- timedelta(hours=simulation_len)
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H') 

    # Extract all unique months between start and end time
    mons = pd.date_range(fp_range_start, fp_range_end, freq='D').strftime("%Y%m").unique().tolist()
    return mons

def footprint_hours(fp_filelist, simulation_len):
    """ Function to extract the hours of the footprint files."""
    # Define time range 
    timestr_start = fp_filelist[0][-37:-24]
    timestr_end = fp_filelist[-1][-37:-24]
    fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')- timedelta(hours=simulation_len)
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H') 

    # Define list of times
    times = pd.date_range(start=fp_range_start, end=fp_range_end, freq='H')

    # Drop times that are not in the range of the footprint files
    for datetime in times:
        if datetime.hour not in range(fp_range_start.hour, (fp_range_end + timedelta(hours=1)).hour):
            times = times.drop(datetime)    
    return times

def find_fluxfiles(fluxdir, variablelist_files, months):
    """ Function to find all fluxfiles in a fluxdir, given a list of variables and months."""
    fluxstring = []
    for var in variablelist_files:
        for mon in months:
            fluxstring += sorted(glob.glob(fluxdir + var + '.' + mon + '.nc'))
    return fluxstring

def open_multiple_fluxfiles(fluxfile_list, variablelist_vars):
    """ Open all files in fluxstring as xr_mfdataset, and sum 
    all the variables in variablelist. """
    ds = xr.open_mfdataset(fluxfile_list, combine='by_coords')
    ds = sum_variables(ds, variablelist_vars)
    return ds

def sum_variables(dset, listofvars):
    """ Function to sum all variables in listofvars in a dataset."""
    data = 0
    for var in listofvars:
        data = data + dset[var][:,:,:] # slice of each variable
    return data 

def auke_add(dset, listofvars):
    return dset[listofvars].sum(axis=0)

def get_3d_station_location(station, stationsfile):
    """ Function to get the lat, lon and agl of a station given the 
    station name and the stationsfile. """
    lat = stationsfile[stationsfile['code']==station]['lat'].values[0]
    lon = stationsfile[stationsfile['code']==station]['lon'].values[0]
    agl = stationsfile[stationsfile['code']==station]['alt'].values[0]
    return lat,lon,agl

def get_latlon_indices(footprint_df, lats, lons):
    """ Function to get the lat and lon indices of the footprint file,
    given the lat and lon lists of the domain. """
    lat_indices = [lats.index(i) for i in list(np.round(footprint_df.variables['Latitude'][:].tolist(), 4))]
    lon_indices = [lons.index(i) for i in list(np.round(footprint_df.variables['Longitude'][:].tolist(), 4))]
    
    lat_indices = xr.DataArray(lat_indices, dims=['pos'])
    lon_indices = xr.DataArray(lon_indices, dims=['pos'])
    return lat_indices, lon_indices

def find_footprint_timeindex(footprint_df, fp_starttime, list_of_months):
    """ Function to get the hours since the start of the footprint file. """
    # Calculate hours since start of CTE-HR flux file to use later for indexing
    time_diff = (fp_starttime - pd.to_datetime(list_of_months[0], format="%Y%m"))
    hours_since_start_of_ds = time_diff.seconds // 3600 + time_diff.days * 24
    hours_into_file = (hours_since_start_of_ds - footprint_df.variables['Time'][:]).astype(int)
    hours_into_file = xr.DataArray(hours_into_file, dims=['pos'])
    return hours_into_file

def get_flux_contribution(flux_dataset, time_index_list, lat_index_list, lon_index_list, footprint_df, infl_varname):
    """ Function to get the flux contribution of a footprint file, with list indexing."""
    flux_contribution_perhr = flux_dataset[time_index_list, lat_index_list, lon_index_list] * footprint_df.variables[infl_varname][:]
    flux_contribution_sum = flux_contribution_perhr.sum()
    return flux_contribution_sum

def create_obs_sim_dict(flux_dataset, fp_filelist, lats, lons, list_of_mons, RDatapath, 
                        bgpath, npars, station_lat, station_lon, station_agl, stationname,
                        save_as_pkl = False):
    """ Function to create a dictionary with the background, mixed and pseudo observation. """
    # Create empty dictionary
    summary_dict = {}

    # Loop over all footprint files
    for i in tqdm(range(0,len(fp_filelist))):
        
        # Get start and end time of footprint
        file = fp_filelist[i]
        footprint_df = nc.Dataset(file, 'r')

        # Get start time of footprint
        timestr = file[-37:-24]
        fp_starttime = pd.to_datetime(timestr, format="%Yx%mx%dx%H")
        #obspack_starttime = int((fp_starttime - obspack_basetime).total_seconds())

        # Extract latitude indices from sparse footprint file and insert into xr.DataArray for later indexing
        lat_indices, lon_indices = get_latlon_indices(footprint_df, lats, lons)        
        hours_into_file = find_footprint_timeindex(footprint_df = footprint_df, fp_starttime = fp_starttime, 
                                                   list_of_months = list_of_mons)

        # Calculate mean background concentration for current footprint
        bg = calculate_mean_bg(fp_starttime = fp_starttime, RDatapath = RDatapath, bgpath = bgpath, 
                               npars = npars, lat = station_lat, lon = station_lon, agl = station_agl)
        
        # Calculate mixed flux contribution to mole fraction signal for current footprint
        mixed = get_flux_contribution(flux_dataset = flux_dataset, time_index_list = hours_into_file,
                                      lat_index_list = lat_indices, lon_index_list = lon_indices, 
                                      footprint_df = footprint_df, infl_varname = 'Influence')

        # Calculate pseudo observation from bg and flux contribution
        pseudo_obs = bg + mixed 
        
        # Create dictionary with all necessary information
        obs_sim_dict = {'time': fp_starttime, 'bg': bg, 'mixed': mixed, 'pseudo_obs': pseudo_obs}

        # Append obs_sim_dict to summary_dict
        summary_dict.append(obs_sim_dict)

    if save_as_pkl == True:
        # Save dictionary as pickle file
        with open('obs_sim_dict_' + stationname + '_' + fp_starttime.yr + '.pkl', 'wb') as f:
            pickle.dump(obs_sim_dict, f)
    elif save_as_pkl == False:
        return summary_dict