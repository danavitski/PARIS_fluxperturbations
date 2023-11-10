# Daan Kivits, 2023

# This file contains functions used by the FLEXPART flux file creation scripts.
# EDIT 15-09-2023: Added functions to use with the STILT model.

##############################################
########## LOAD NECCESSARY PACKAGES ##########
##############################################
from dateutil.relativedelta import relativedelta
from itertools import groupby
import csv
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
from tqdm import tqdm
import logging
import matplotlib.ticker as mtick
import dask
import dask.array as da
from multiprocessing import current_process

def median2d(arr, new_shape):
    """ Function to average any given shape, which is a multiple of the domain size, to the domain size
    Input:
        arr: np.ndarray: original array to be averaged
        new_shape: tuple: shape to be averaged to
    Returns:
        np.ndarray: averaged arr"""
    shape = (new_shape[0], arr.shape[-2] // new_shape[-2],
             new_shape[1], arr.shape[-1] // new_shape[-1])
    if len(arr.shape) == 3:  # If time is included:
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

    src = nc.Dataset(src_file, mode='r')

    # Create a time variable to loop over
    timevar = src.variables['time'][:]
    fluxname = list(src.variables.keys())[3]

    for timeindex, time in enumerate(timevar):
        trg = nc.Dataset(trg_file, mode='r+')

        # Loop over variables of interest and put in lu_specific fluxsets of different year
        if fluxname in variablelist:
            var = trg.variables[fluxname][timeindex, :, :]
            luvar = src.variables[fluxname][timeindex, :, :]

            lu = median2d(lu, luvar.shape[1:])

            var[lu == lu_type] = luvar[lu == lu_type]
            trg.variables[fluxname][timeindex, :, :] = var
            dif = trg.variables[fluxname][timeindex, :, :] - \
                src.variables[fluxname][timeindex, :, :]

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

    src = nc.Dataset(src_file, mode='r')

    # Create a time variable to loop over
    timevar = src.variables['time'][:]
    fluxname = list(src.variables.keys())[3]

    for timeindex, time in enumerate(timevar):
        trg = nc.Dataset(trg_file, mode='r+')

        # Loop over variables of interest and put in lu_specific fluxsets of different year
        if fluxname in variablelist:
            var = trg.variables[fluxname][timeindex, :, :]
            luvar = src.variables[fluxname][timeindex, :, :]

            lu = median2d(lu, luvar.shape[1:])

            var[lu == lu_type] = luvar[lu == lu_type]
            trg.variables[fluxname][timeindex, :, :] = var
            dif = trg.variables[fluxname][timeindex, :, :] - \
                src.variables[fluxname][timeindex, :, :]

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

    # Create a time variable to loop over
    timevar = np.arange(0, len(src.variables['time'][:]), 1)
    time_array = np.arange(0, len(timevar), 1)

    for name, var in src.variables.items():
        if name in variablelist:
            for time in time_array:
                data = src.variables[name][time, :, :]

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
    trg.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

    # Create the variables in the file
    for name, var in src.variables.items():
        trg.createVariable(name, var.dtype, var.dimensions)

        # Copy the variable attributes
        trg.variables[name].setncatts(
            {a: var.getncattr(a) for a in var.ncattrs()})

        # Copy the variables values (as 'f4' eventually)
        trg.variables[name][:] = src.variables[name][:]

    # Save the file
    trg.close()


def date_range(start, end, intv):
    """ Function to split a certain date range according to a certain interval,
    and yield the start and end date of each interval. To put results in a list, 
    do: list(date_range(start, end, intv)).
    """
    start = datetime.strptime(start,"%Y-%m-%d")
    end = datetime.strptime(end,"%Y-%m-%d")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y-%m-%d")
    yield end.strftime("%Y-%m-%d")


def filter_files_by_date(file_list, start_date, end_date):
    """ Function to filter a list of files by a certain date range. 
    """
    filtered_files = []

    date_format = "%Y-%m-%d"  # adjust the format based on your actual filenames

    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    for file_name in file_list:
        date_str = file_name.split("_")[2][:13]  # adjust the index based on your actual filenames
        file_datetime = datetime.strptime(date_str, "%Yx%mx%dx%H")

        if start_datetime <= file_datetime <= end_datetime:
            filtered_files.append(file_name)

    return filtered_files


def parse_datetime(datetime_str):
    """ Function to parse a datetime string from a netcdf file to a datetime object. 
    NOTE: only the first 19 characters are parsed, because the netcdf files often contain
    a string with the format 'YYYY-MM-DDTHH:MM:SS' and some extra unwanted characters.
    
    """
    return datetime.strptime(datetime_str[0:19].decode('utf-8'), '%Y-%m-%dT%H:%M:%SZ')


def coordinate_list(lat_ll, lat_ur, lon_ll, lon_ur, lat_step, lon_step):
    """ Function to create a list of lons and lats given a 
    range and stepsize.
    """
    lats = list(np.round(np.arange(lat_ll, lat_ur, lat_step), 4))
    lons = list(np.round(np.arange(lon_ll, lon_ur, lon_step), 4))
    return lats, lons


def footprint_unique_months(fp_filelist, simulation_len):
    """ Function to extract the unique months of the footprint files, to later use to subset CTE-HR flux files (time format YYYYmm). 
    The function accounts for extra possible term in filename, that describes the ensemble run number.
    
    """
    # Define time range
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_start = fp_filelist[0][-42:-29]
        timestr_end = fp_filelist[-1][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_start = fp_filelist[0][-45:-32]
        timestr_end = fp_filelist[-1][-45:-32]
    
    fp_range_start = datetime.strptime(
        timestr_start, '%Yx%mx%dx%H') - timedelta(hours=simulation_len)
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H')

    # Extract all unique months between start and end time
    mons = pd.date_range(fp_range_start, fp_range_end,
                         freq='D').strftime("%Y%m").unique().tolist()
    return mons


def footprint_hours(fp_filelist, simulation_len):
    """ Function to extract the hours of the footprint files.
        The function accounts for extra possible term in filename, that describes the ensemble run number.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_start = fp_filelist[0][-42:-29]
        timestr_end = fp_filelist[-1][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_start = fp_filelist[0][-45:-32]
        timestr_end = fp_filelist[-1][-45:-32]
    
    # Define time range
    fp_range_start = datetime.strptime(
        timestr_start, '%Yx%mx%dx%H') - timedelta(hours=simulation_len)
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H')

    # Define list of times
    times = pd.date_range(start=fp_range_start, end=fp_range_end, freq='H')

    # Drop times that are not in the range of the footprint files
    for datetime in times:
        if datetime.hour not in range(fp_range_start.hour, (fp_range_end + timedelta(hours=1)).hour):
            times = times.drop(datetime)

    return times


def footprints_extract_starttime(fp_filelist):
    """ Function to extract the start and end time of the footprint files.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_start = fp_filelist[0][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_start = fp_filelist[0][-45:-32]

    fp_datetime_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')

    return fp_datetime_start


def footprints_extract_endtime(fp_filelist):
    """ Function to extract the start and end time of the footprint files.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_end = fp_filelist[-1][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_end = fp_filelist[-1][-45:-32]
    fp_datetime_end = pd.to_datetime(timestr_end, format = '%Yx%mx%dx%H')

    return fp_datetime_end


def footprint_extract_npars(fp_file):
    """ Function to extract the number of particles in the footprint files.
    """
    # Define time string
    if len(fp_file.split(sep='x')) == 8:
        npars = int(fp_file.split(sep='x')[-1].split(sep='.')[0].strip())
    elif len(fp_file.split(sep='x')) == 9:
        npars = int(fp_file.split(sep='x')[-2].strip())

    return npars


def footprint_extract_time(fp_file):
    """ Function to extract the start and end time of the footprint files.
    """
    # Define time string
    if len(fp_file.split(sep='x')) == 8:
        timestr = fp_file[-42:-29]
    elif len(fp_file.split(sep='x')) == 9:
        timestr = fp_file[-45:-32]
    fp_datetime = pd.to_datetime(timestr, format = '%Yx%mx%dx%H')

    return fp_datetime


def find_fluxfiles(fluxdir, fluxtype, perturbationcode=None, variablelist_files=None, months=None):
    """ Function to find all fluxfiles in a fluxdir, given a list of variables and months.
    Choose between multiple fluxtypes, currently still limited to "CTEHR" and "PARIS" (03-11-2023)."""
    fluxstring = []
    
    if fluxtype == "PARIS":
        fluxstring += sorted(glob.glob(fluxdir + '/' + perturbationcode + '/paris_ctehr_*.nc'))
    
    elif fluxtype == "CTEHR":
        for var in variablelist_files:
            for mon in months:
                fluxstring += sorted(glob.glob(fluxdir + var + '.' + mon + '.nc'))
    else:
        print('Input fluxtype not recognized, please select one that is defined in the find_fluxfiles() function or define one yourself.')

    return fluxstring


def open_fluxfiles(fluxfile_list, start_date=None, end_date=None, fluxtype=None, sum=False, variablelist_vars=None):
    """ Function to open all files in fluxstring as xr_mfdataset, and sum 
    all the variables in variablelist. Choose between multiple fluxtypes, currently still limited to 
    "CTEHR" and "PARIS" (03-11-2023)."""
    
    if fluxtype == "PARIS":        
        # Load fluxfiles into one mfdataset 
        ds = xr.open_mfdataset(fluxfile_list, chunks={'time': -1, 'lat': -1, 'lon': -1})
        
        if start_date != None and end_date != None:
            ds = ds.sel(time=slice(start_date, end_date))

        # Option to sum over all variables, or not!
        if sum==True:
            ds = sum_variables(ds, variablelist_vars)
        
        # Subset xarray ds only neccessary variables
        return ds[variablelist_vars]

    elif fluxtype == "CTEHR":
        # Load fluxfiles into one mfdataset 
        ds = xr.open_mfdataset(fluxfile_list, combine='by_coords', chunks={'time': -1, 'lat': -1, 'lon': -1})
    
        # Option to sum over all variables, or not!
        if sum==True:
            ds = sum_variables(ds, variablelist_vars)
        
        # Subset xarray ds only neccessary variables
        return ds[variablelist_vars]

    else:
        print('Input fluxtype not recognized, please select one that is defined in the find_fluxfiles() function or define one yourself.')


def sum_variables(dset, listofvars):
    """ Function to sum all variables in listofvars in a dataset."""
    data = 0
    for var in listofvars:
        data = data + dset[var][:, :, :]  # slice of each variable
    return data


def get_3d_station_location(stationsfile, station=str(), colname_lon=str(), colname_lat=str(), colname_agl=str()):
    """ Function to get the lat, lon and agl of a station given the 
    station name and the stationsfile. """
    lat = stationsfile[stationsfile['code'] == station][colname_lat].values[0]
    lon = stationsfile[stationsfile['code'] == station][colname_lon].values[0]
    agl = stationsfile[stationsfile['code'] == station][colname_agl].values[0]

    if (np.isnan(agl)):
        agl = stationsfile[stationsfile['code'] == station]['alt'].values[0]

    return lat, lon, agl


def get_latlon_indices(footprint_df, lats, lons):
    """ Function to get the lat and lon indices of the footprint file,
    given the lat and lon lists of the domain. """
    lat_indices = [lats.index(i) for i in list(
        np.round(footprint_df.variables['Latitude'][:].tolist(), 4))]
    lon_indices = [lons.index(i) for i in list(
        np.round(footprint_df.variables['Longitude'][:].tolist(), 4))]

    lat_indices = xr.DataArray(lat_indices, dims=['pos'])
    lon_indices = xr.DataArray(lon_indices, dims=['pos'])
    return lat_indices, lon_indices


def find_footprint_timeindex(footprint_df, fp_starttime, fp_filelist):
    """ Function to get the hours since the start of the footprint file. """
    fp_filelist_starttime = footprints_extract_starttime(fp_filelist)
    
    # Calculate hours since start of CTE-HR flux file to use later for indexing
    time_diff = (fp_starttime - fp_filelist_starttime)
    hours_since_start_of_ds = time_diff.seconds // 3600 + time_diff.days * 24
    hours_into_file = (hours_since_start_of_ds -
                       footprint_df.variables['Time'][:]).astype(int)
    hours_into_file = xr.DataArray(hours_into_file, dims=['pos'])
    return hours_into_file


def get_flux_contribution_mixed(flux_dataset, time_index_list, lat_index_list, lon_index_list, footprint_df, infl_varname):
    """ Function to get the MIXED flux contribution of a footprint file, with list indexing."""
    
    flux_contribution_perhr = flux_dataset[time_index_list, lat_index_list,
                                        lon_index_list] * footprint_df.variables[infl_varname][:]
    flux_contribution_sum = flux_contribution_perhr.sum().compute().values
    return flux_contribution_sum


def get_flux_contribution_ss(flux_dataset, time_index_list, lat_index_list, lon_index_list, footprint_df, infl_varname, variablelist_vars):
    """ Function to get the SECTOR-SPECIFIC flux contribution of a footprint file, with list indexing."""
    
    delayed_results = [dask.delayed(flux_dataset[v][time_index_list, lat_index_list, lon_index_list] * footprint_df.variables[infl_varname][:]).sum() for v in variablelist_vars]
    sumlist = [result.item() for result in dask.compute(*delayed_results)]

    return sumlist


def create_obs_sim_dict_ens(flux_dataset, fp_filelist, lats, lons, RDatapath,
                        bgpath, station_lat, station_lon, station_agl, stationname,
                        save_as_pkl=False):
    """ Function to compare the STILT simulation results from simulations with different setups, e.g. a
    sensitivity analysis. Here, the sensitivity analysis is done on the amount of particles released in 
    each of the STILT simulations, with multiple ensemble members to increase statistical significance.
    
    NOTE: Not up to date!
    """
    logging.info('Current station is ' + stationname + ', with lat = ' + str(station_lat) + ', lon = ' + str(station_lon) + ', and agl = ' + str(station_agl))
    logging.info('Starting calculation of flux contributions and pseudo observations.')
    
    # Sort footprint list (required by groupby function)
    fp_filelist.sort()

    # Group subgroup by released number of particles
    fp_filelist_grouped = [list(g) for k, g in groupby(fp_filelist, lambda s: s[-4:-5])]
    print(fp_filelist_grouped)

    # Create empty df to store results
    pseudo_df, mixed_df = (pd.DataFrame() for i in range(2))
    
    for time_specific_list in fp_filelist_grouped:  
        time_specific_multilist = [list(g) for k, g in groupby(time_specific_list, lambda s: s[-9:-6])]
        print(time_specific_multilist)

        for part_specific_list in time_specific_multilist:
            print(part_specific_list)
            
            # Create empty dictionary
            pseudodict, mixeddict = ({} for i in range(2))
            npars = int(part_specific_list[0].split(sep='x')[-2].strip())

            logging.info('Starting up with ' + str(npars) + ' particles.')

            # Loop over all footprint files
            for i in tqdm(range(0, len(part_specific_list))):
                # Get start and end time of footprint
                file = part_specific_list[i]
                footprint_df = nc.Dataset(file, 'r')

                # Get start time and npars of footprint
                npars = footprint_extract_npars(file)
                fp_starttime = footprint_extract_time(file)
                
                logging.info(f'Current ens_mem_num is {i+1}')
                
                # Extract latitude indices from sparse footprint file and insert into xr.DataArray for later indexing
                lat_indices, lon_indices = get_latlon_indices(footprint_df, lats, lons)
                hours_into_file = find_footprint_timeindex(footprint_df=footprint_df, fp_starttime=fp_starttime,
                                                        fp_filelist = fp_filelist)

                logging.info(f'This footprint has {len(lat_indices)} values, and ran for {len(np.unique(hours_into_file))} hours backwards in time')

                # Calculate mean background concentration for current footprint
                bg = calculate_mean_bg_ens(fp_starttime=fp_starttime, RDatapath=RDatapath, bgpath=bgpath,
                                    npars=npars, lat=station_lat, lon=station_lon, agl=station_agl, ens_mem_num=i+1)

                logging.info(f'Mean background is {bg}')

                mixed = get_flux_contribution_mixed(flux_dataset=flux_dataset, time_index_list=hours_into_file,
                                            lat_index_list=lat_indices, lon_index_list=lon_indices,
                                            footprint_df=footprint_df, infl_varname='Influence')
                
                logging.info(f'Mixed flux contribution is {mixed}')

                # Calculate pseudo observation from bg and flux contribution
                pseudo_obs = bg + mixed

                # Save keys and values in summary_dict
                pseudodict[i+1] = [pseudo_obs]
                mixeddict[i+1] = [mixed]

            pseudo_subdf = pd.DataFrame(pseudodict, index=[npars])
            pseudo_subdf.index.name = 'Number of particles'
            pseudo_df = pseudo_df.append(pseudo_subdf)

            mixed_subdf = pd.DataFrame(mixeddict, index=[npars])
            mixed_subdf.index.name = 'Number of particles'
            mixed_df = mixed_df.append(mixed_subdf)

            print('Pseudo df: ' + str(pseudo_df))
            print('Mixed df: ' + str(mixed_df))

    logging.info('Finished calculating flux contributions and pseudo observations.')
        
    return pseudo_df, mixed_df
        

def create_obs_sim_dict(flux_dataset, fp_filelist, lats, lons, RDatapath,
                        bgpath, npars, station_lat, station_lon, station_agl, stationname,
                        variablelist_vars, stilt_rundir, save_as_pkl=False):
    """ Function to create a dictionary with the background, mixed and pseudo observation. 
    
    """
    # Create empty dictionary
    summary_dict = {}
    
    logging.info('Current station is ' + stationname + ', with lat = ' + str(station_lat) + ', lon = ' + str(station_lon) + ', and agl = ' + str(station_agl))
    logging.info('Starting calculation of flux contributions and pseudo observations.')

    # Import list of missing footprint files
    missing_fp_filestr = stilt_rundir + stationname + '/missing.footprints'
    if os.path.isfile(missing_fp_filestr):
        missing_fp_file = pd.read_csv(missing_fp_filestr)
        missing_fp_filelist = list(missing_fp_file['ident'])

    # Loop over all footprint files
    for i in tqdm(range(0, len(fp_filelist))):

        # Get start and end time of footprint
        file = fp_filelist[i]
        ident = os.path.splitext(os.path.basename(file).split(sep='_')[-1])[0]
        
        # Get start time and npars of footprint
        npars = footprint_extract_npars(file)
        fp_starttime = footprint_extract_time(file)

        logging.info(f'Starttime of current footprint is {fp_starttime}')
        
        # Create keys and values for summary_dict later
        key = fp_starttime.to_numpy()

        # Calculate mean background concentration for current footprint
        bg = calculate_mean_bg(fp_starttime=fp_starttime, RDatapath=RDatapath, bgpath=bgpath,
                    npars=npars, lat=station_lat, lon=station_lon, agl=station_agl)

        logging.info(f'Mean background is {bg}')

        if os.path.exists(missing_fp_filestr)==False or ident not in missing_fp_filelist:
            # Load footprint as netCDF4 dataset
            footprint_df = nc.Dataset(file, 'r')

            # Extract latitude indices from sparse footprint file and insert into xr.DataArray for later indexing
            lat_indices, lon_indices = get_latlon_indices(footprint_df, lats, lons)
            hours_into_file = find_footprint_timeindex(footprint_df=footprint_df, fp_starttime=fp_starttime,
                                                    fp_filelist=fp_filelist)

            logging.info(f'This footprint has {len(lat_indices)} values, and ran for {len(np.unique(hours_into_file))} hours backwards in time')

            # Calculate CTE-HR contribution
            mixed = get_flux_contribution_ss(flux_dataset=flux_dataset, time_index_list=hours_into_file,
                                        lat_index_list=lat_indices, lon_index_list=lon_indices,
                                        footprint_df=footprint_df, infl_varname='Influence',
                                        variablelist_vars=variablelist_vars)
        elif ident in missing_fp_filelist:
            logging.info(f'Footprint is empty and no surface exchange with particles is recorded, taking background concentration at station location as pseudo observation.')
            mixed = [0] * len(variablelist_vars)

        # Calculate pseudo observation from bg and flux contribution            
        if type(mixed) != int:
            contribution_mixed = sum(mixed)
            logging.info(f'Mixed flux contribution is {contribution_mixed}')

            pseudo_obs = bg + contribution_mixed
            values = mixed + [bg, contribution_mixed, pseudo_obs]
        else:
            logging.info(f'Mixed flux contribution is {mixed}')           
            
            pseudo_obs = bg + mixed
            values = [bg, mixed, pseudo_obs]
           
        # Save keys and values in summary_dict
        summary_dict[key] = values

    logging.info('Finished calculating flux contributions and pseudo observations.')

    if save_as_pkl == True:
        logging.info('Saving dictionary as pickle file')

        # Save dictionary as pickle file
        with open('obs_sim_dict_' + stationname + '_' + fp_starttime.yr + '.pkl', 'wb') as f:
            pickle.dump(summary_dict, f)
    
    elif save_as_pkl == False:
        logging.info('Saving dictionary in Python memory.')

        return summary_dict

def write_dict_to_ObsPack(obspack_list, obs_sim_dict, variablelist_vars, stationcode, outpath='/projects/0/ctdas/PARIS/DATA/obspacks/'):
    """ Function to write a dictionary (that contains the simulation results, the background,
    and pseudo_observations) to a copy of the original ObsPacks in the ObsPack collection.

    Input variables:
    obspack_list: list of ObsPacks that meet the filtering criteria
    obs_sim_dict: dictionary that contains the simulation results, the background, and observations
    stationcode: stationcode of the station
    outpath: path to the output directory
    """
    # Find ObsPack file in list of ObsPacks
    obspack_orig = [s for s in obspack_list if stationcode.lower() in s.lower()][0]
    if obspack_orig is None:
        print(f"No ObsPack found for station code {stationcode}")
        return

    # Define new obspack filename
    obspack = os.path.join(outpath, 'pseudo_' + os.path.basename(obspack_orig))

    # Copy ObsPack file to newfile location to at a fill later stage
    #logging.info(f'Copying ObsPack file from {obspack_orig} to {obspack}')
    #shutil.copy(obspack_orig, obspack)

    # Define time since epoch, which is also the start time of the ObsPack file
    obspack_basetime = np.datetime64('1970-01-01T00:00:00')

    # Calculate start-time of simulation in seconds since epoch
    #fp_starttime = min(obs_sim_dict)
    #fp_endtime =  max(obs_sim_dict)
    #min_time = (fp_starttime - obspack_basetime) / np.timedelta64(1, 's')
    #max_time = (fp_endtime - obspack_basetime) / np.timedelta64(1, 's')
    
    # Open newfile in 'append' mode
    with xr.open_dataset(obspack_orig, decode_times=False) as ds:
        
        # Resample to hourly data
        logging.info('Resampling to hourly data')
        ds['time'] = pd.to_datetime(ds['time'], unit='s')
        logging.info(ds['time'])

        # Copy variable 'value' to new variables to fill later
        new_variables = variablelist_vars + ['mixed', 'background', 'pseudo_observation']
        for new_variable_name in new_variables:
            dummy_var = xr.DataArray(
                name=new_variable_name,
                dims=ds.time.dims,
                coords=ds.coords
            )
            ds[new_variable_name] = dummy_var
        
        # Select only the time range of the simulation (add 3600 seconds to max_time to include last hour)
        #subds = ds.sel(time=slice(min_time, max_time + 3600))
        subds = ds.sel(time=slice(datetime(2021,1,1), datetime(2022,1,1)))
        non_numeric_vars = [var for var in subds.variables if subds[var].dtype not in ['float64', 'int64','float32','int32', 'int8']][:]

        #subds_rs1 = subds['value'].resample(time='1H').nearest(tolerance=None)
        subds_rs1 = subds['value'].resample({'time':'H'}).mean(dim='time', skipna=True, keep_attrs=True).dropna(dim='time')
        #subds_rs = subds['value'].resample(time='1H').mean(dim='time', keep_attrs=True)
        
        #subds_rs2 = subds
        #subds_rs2.coords['time'] = subds.time.dt.floor('1H')  
        #subds_rs2 = subds_rs2.groupby('time').mean()
        
        #missing_time_values = set(subds_rs1['time'].values) - set(subds_rs2['time'].values)
        #logging.info(len(missing_time_values))
        #logging.info(missing_time_values)

        
        #logging.info(subds['value'].sel(time='2021-12-02T18:00:00.000000000', method='nearest'))
        #logging.info(subds_rs1.sel(time='2021-12-02T18:00:00.000000000', method='nearest'))
        #logging.info(subds_rs2['value'].sel(time='2021-12-02T18:00:00.000000000', method='nearest'))
        #logging.info(non_numeric_vars)
        #logging.info(subds_rs1[non_numeric_vars[1]])
        #logging.info(subds_rs2[non_numeric_vars[1]])

        # Change time and value variables to resampled counterparts
        subds['time'] = subds_rs1['time']
        subds['value'] = subds_rs1
        
        #### NOTE: the non-numerical ObsPack variables are thrown out during the resampling process,
        #### so they have to be resampled manually.
        # Manually resample ObsPack start_time element
        subds[non_numeric_vars[0]] = subds[non_numeric_vars[0]] - np.timedelta64(30, 'M')
        #subds[non_numeric_vars[0]] = subds[non_numeric_vars[0]] - timedelta(minutes=30)

        # Manually resample ObsPack datetime element
        subds[non_numeric_vars[1]] = xr.apply_ufunc(parse_datetime, subds[non_numeric_vars[1]], vectorize=True) - timedelta(minutes=30)

        # Calculate time differences only once
        #ds_time_seconds = ((subds['time'] - obspack_basetime).values.astype('timedelta64[s]')).astype(np.int64)
        ds_time_seconds = ((subds['time'] - obspack_basetime).values.astype('int64')) / 1e9

        # Loop over all keys and values in dictionary
        logging.info(f'Looping over {len(obs_sim_dict)} footprints')

        for key, values in obs_sim_dict.items():
            logging.info(f'Current footprint time is {key}')

            # Calculate time in seconds since start of ObsPack file
            obspack_time = int(((key - obspack_basetime) / np.timedelta64(1, 's')).item())
            
            # Calculate time difference between simulation time and ObsPack time
            diff_dt = obspack_time - ds_time_seconds
            diff_index = np.abs(diff_dt).argmin()
            #diff_min = diff_dt[diff_index]

            # Convert values to numpy array
            values = np.array(values)

            # Write values to new variables in new ObsPack file
            for v in new_variables:
                subds[v][diff_index] = values[variablelist_vars.index(v)]

    # Save and close the new file
    logging.info(f'Writing results to {obspack}')
    subds.to_netcdf(obspack, mode='w')

def main(filepath, sim_length, fluxdir, stilt_rundir, fluxvarnamelist, perturbationcode, stationsfile, stationcode,
         outpath, bgfilepath, npars, obspack_list, lats, lons, verbose, fluxtype, date_pair):
    """ Main function to run the footprint - flux multiplication script. """

    if verbose == True:
        logging.basicConfig(level=logging.INFO, format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.CRITICAL)

    # Get 3D station location
    lat,lon,agl = get_3d_station_location(stationsfile, stationcode, colname_lat='lat', colname_lon='lon',
                                            colname_agl='corrected_alt')  

    # Get list of footprint files for station
    sparse_files = sorted(glob.glob(filepath + 'footprint_' + stationcode + '_2021x01x01x14*.nc'))

    if fluxtype == 'PARIS':
        # Only once per station, create list of all PARIS flux files
        fluxstring = find_fluxfiles(fluxdir = fluxdir, perturbationcode=perturbationcode, fluxtype=fluxtype)
        
        # Filter list of footprint files by date range (using start_date and end_date, if they exist!)
        if date_pair != None:
            start_date, end_date = date_pair
            sparse_files = filter_files_by_date(sparse_files, start_date, end_date)

            # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
            cte_ds = open_fluxfiles(fluxstring, start_date = start_date, end_date = end_date, variablelist_vars = fluxvarnamelist, fluxtype=fluxtype)
        else:
            # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
            cte_ds = open_fluxfiles(fluxstring, variablelist_vars = fluxvarnamelist, fluxtype=fluxtype)

        # Loop over all footprints files
        obs_sim_dict = create_obs_sim_dict(fp_filelist = sparse_files, flux_dataset = cte_ds,
                            lats = lats, lons = lons, RDatapath = filepath, 
                            bgpath = bgfilepath, npars = npars, station_lat = lat, station_lon = lon, 
                            station_agl = agl, stationname = stationcode, variablelist_vars = fluxvarnamelist,
                            stilt_rundir = stilt_rundir, save_as_pkl=False)
                            
    elif fluxtype == 'CTEHR':
        # Define which CTE-HR variables to find fluxfiles for
        fluxfilenamelist = ['nep', 'fire', 'ocean', 'anthropogenic']

        # Extract all unique months between start and end time
        mons = footprint_unique_months(sparse_files, sim_length)

        # Only once per station, create list of all CTE-HR flux files
        fluxstring = find_fluxfiles(fluxdir = fluxdir, variablelist_files = fluxfilenamelist, months = mons, fluxtype=fluxtype)

        # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
        cte_ds = open_fluxfiles(fluxstring, variablelist_vars = fluxvarnamelist, fluxtype=fluxtype)  
 
        # Loop over all footprints files
        obs_sim_dict = create_obs_sim_dict(fp_filelist = sparse_files, flux_dataset = cte_ds,
                            lats = lats, lons = lons, list_of_mons = mons, RDatapath = filepath, 
                            bgpath = bgfilepath, npars = npars, station_lat = lat, station_lon = lon, 
                            station_agl = agl, stationname = stationcode, variablelist_vars = fluxvarnamelist,
                            stilt_rundir = stilt_rundir, save_as_pkl=False)

    write_dict_to_ObsPack(obspack_list = obspack_list, obs_sim_dict = obs_sim_dict, 
                          variablelist_vars=fluxvarnamelist, stationcode = stationcode, outpath = outpath)
    
    logging.info(f'Finished writing results to ObsPack file for station {stationcode}')

def compare_ens_members(filepath, sim_length, fluxdir, fluxfilenamelist, fluxvarnamelist, stationsfile,
        bgfilepath, lats, lons, stationcode):
    
    # Get list of footprint files for station
    sparse_files = sorted(glob.glob(filepath + 'footprint_' + stationcode + '_2021x07x05x17*.nc'))

    # Extract all unique months between start and end time
    mons = footprint_unique_months(sparse_files, sim_length)

    # Only once per station, create list of all CTE-HR flux files
    fluxstring = find_fluxfiles(fluxdir = fluxdir, variablelist_files = fluxfilenamelist, months = mons, fluxtype='CTE-HR')

    # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
    cte_ds = open_fluxfiles(fluxstring, variablelist_vars = fluxvarnamelist)

    # Get 3D station location
    lat,lon,agl = get_3d_station_location(stationsfile, stationcode, colname_lat='lat', colname_lon='lon',
                                            colname_agl='corrected_alt')      
 
    # Loop over all footprints files
    pseudo_df, mixed_df = create_obs_sim_dict_ens(fp_filelist = sparse_files, flux_dataset = cte_ds,
                        lats = lats, lons = lons, list_of_mons = mons, RDatapath = filepath, 
                        bgpath = bgfilepath, station_lat = lat, station_lon = lon, 
                        station_agl = agl, stationname = stationcode)
    
    return pseudo_df, mixed_df
    
def plot_ens_members_fromdf(df):
    """ Function to plot the results of the ensemble members from a dataframe in memory. """
    # Create subplots
    fig, axes = plt.subplots(len(df.index), 1, figsize=(15, 5), sharey=True)
    axes[0].set_title('Simulation results for varying amounts of particles')
    
    for i in range(0, len(df.index)):
        npars = df.index[i]
        ax = axes[i]
        df.loc[i].plot(ax=ax, marker='o', label=str(npars) + 'particle simulation')

        ax.set_xlabel('Number of particles')
        ax.set_ylabel('CO2 amount fraction (-)')
        ax.legend()
        ax.grid()
    
    plt.show()

def plot_ens_members_fromcsv(pseudo_csv, mixed_csv, save = True):
    """ Function to visualize the results of STILT simulations with a varying number
    of particles that have been released. The flux contribution is shown next to 
    the total simulated amount fraction. Both of these are read in separately through
    .CSV files. 
    
    """
    # Read in CSV as pd.DataFrame
    pseudo_df = pd.read_csv(pseudo_csv, index_col="Number of particles")
    mixed_df = pd.read_csv(mixed_csv, index_col="Number of particles")

    # create tickslist
    #tickslist = list(range(100, 400, 100))
    tickslist = [100, 200, 300, 400, 500, 600]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax1 = axes[0]
    ax2 = axes[1]
    ax1.yaxis.get_offset_text().set_fontsize(12)  # Adjust font size if needed

    ax1.set_title('Total simulated mixing ratio \n (pseudo-observations)', pad = 10)
    ax1.set_xlabel('Number of particles')
    ax1.set_ylabel('CO2 amount fraction [ppm]')
    ax1.set_ylim(410,415)
    ax1.set_xticks(tickslist)
    ax1.set_xticklabels(pseudo_df.index)
    ax1.grid()

    ax2.set_title('CTE-HR contribution to total \nCO2 amount fraction', pad = 10)
    ax2.set_ylabel('CTE-HR contribution to total \nCO2 amount fraction [ppm]')
    ax2.set_xlabel('Number of particles')
    ax2.set_ylim(-1,-6)
    ax2.set_xticks(tickslist)
    ax2.set_xticklabels(mixed_df.index)
    ax2.grid()
    ax2.invert_yaxis()

    plt.subplots_adjust(wspace = 0.3)
    
    for i in range(0, len(pseudo_df.index)):
        npars = pseudo_df.index[i]
        ax1.scatter(x=[tickslist[i]] * len(pseudo_df.columns), y=pseudo_df.iloc[i]*1e6, marker='o', c='orange', label=str(npars) + 'particle simulation', s = 10, alpha = 1, zorder=3) 
        ax2.scatter(x=[tickslist[i]] * len(mixed_df.columns), y=mixed_df.iloc[i]*1e6, marker='s', c='red', label=str(npars) + 'particle simulation', s = 10, alpha = 1, zorder = 3)
    
    if (save == True):
        plt.savefig('/projects/0/ctdas/PARIS/Experiments/ens_members_comparison/ens_members_CBW207.png', dpi=300, bbox_inches='tight', overwrite=True)

    plt.show()
    plt.close()
