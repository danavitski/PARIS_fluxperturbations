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
        date_str = file_name.split("_")[3][:13]  # adjust the index based on your actual filenames
        file_datetime = datetime.strptime(date_str, "%Yx%mx%dx%H")

        if start_datetime <= file_datetime <= end_datetime:
            filtered_files.append(file_name)

    return filtered_files


def parse_datetime(datetime_dataarray):
    """ Function to parse a datetime string from a netcdf file to a datetime object. 
    NOTE: only the first 19 characters are parsed, because the netcdf files often contain
    a string with the format 'YYYY-MM-DDTHH:MM:SS' and some extra unwanted characters.
    
    """
    #return pd.to_datetime(datetime_str[0:19].decode('utf-8'), '%Y-%m-%dT%H:%M:%S')
    datetime_str = datetime_dataarray.astype(str).str[0:19]
    return pd.to_datetime(datetime_str, format='%Y-%m-%dT%H:%M:%S')


def coordinate_list(lat_ll, lat_ur, lon_ll, lon_ur, lat_step, lon_step):
    """ Function to create a list of lons and lats given a 
    range and stepsize.
    """
    #lats = list(np.round(np.arange(lat_ll + 0.5 * lat_step, lat_ur - 0.5 * lat_step, lat_step), 4))
    #lons = list(np.round(np.arange(lon_ll + 0.5 * lon_step, lon_ur - 0.5 * lon_step, lon_step), 4))
    lats = np.array(np.round(np.arange(lat_ll + 0.5 * lat_step, lat_ur - 0.5 * lat_step, lat_step), 4))
    lons = np.array(np.round(np.arange(lon_ll + 0.5 * lon_step, lon_ur - 0.5 * lon_step, lon_step), 4))
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


def footprint_hours(fp_filelist, simulation_len, shift_forward=False):
    """ Function to extract the hours of a list of footprint files.
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
    if shift_forward:
        fp_range_start = datetime.strptime(
            timestr_start, '%Yx%mx%dx%H') - timedelta(hours=simulation_len)
    else:
        fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')
    
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H')

    # Define list of times
    times = pd.date_range(start=fp_range_start, end=fp_range_end, freq='H')

    # Drop times that don't have the same Hour of Day (HOD) as the footprint files
    for time in times:
        if time.hour not in range(fp_range_start.hour, (fp_range_end + timedelta(hours=1)).hour):
            times = times.drop(time)

    return times.tolist()


def footprint_list_extract_starttime(fp_filelist):
    """ Function to extract the start time of A LIST OF footprint files.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_start = fp_filelist[0][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_start = fp_filelist[0][-45:-32]

    fp_datetime_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')

    return fp_datetime_start


def footprint_list_extract_endtime(fp_filelist):
    """ Function to extract the end time of A LIST OF footprint files.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        timestr_end = fp_filelist[-1][-42:-29]
    elif len(fp_filelist[0].split(sep='x')) == 9:
        timestr_end = fp_filelist[-1][-45:-32]
    
    fp_datetime_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H')

    return fp_datetime_end


def footprint_extract_starttime(fp_file):
    """ Function to extract the start time of A SPECIFIC footprint file.
    """
    # Define time string
    if len(fp_file.split(sep='x')) == 8:
        timestr = fp_file[-42:-29]
    elif len(fp_file.split(sep='x')) == 9:
        timestr = fp_file[-45:-32]
    #fp_datetime = pd.to_datetime(timestr, format = '%Yx%mx%dx%H')

    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    
    return fp_starttime


def footprint_extract_endtime(fp_file, simulation_len=240):
    """ Function to extract the time range of A SPECIFIC footprint file.
    """
    # Define time string
    if len(fp_file.split(sep='x')) == 8:
        timestr = fp_file[-42:-29]
    elif len(fp_file.split(sep='x')) == 9:
        timestr = fp_file[-45:-32]
    #fp_datetime = pd.to_datetime(timestr, format = '%Yx%mx%dx%H')
    
    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    fp_endtime = fp_starttime - timedelta(hours=simulation_len) 

    return fp_endtime


def footprint_extract_npars(fp_file):
    """ Function to extract the number of particles in the footprint files.
    """
    # Define time string
    if len(fp_file.split(sep='x')) == 8:
        npars = int(fp_file.split(sep='x')[-1].split(sep='.')[0].strip())
    elif len(fp_file.split(sep='x')) == 9:
        npars = int(fp_file.split(sep='x')[-2].strip())

    return npars


def footprint_list_extract_npars(fp_filelist):
    """ Function to extract the number of particles from a list of footprint files.
    """
    # Define time string
    if len(fp_filelist[0].split(sep='x')) == 8:
        npars = int(fp_filelist[0].split(sep='x')[-1].split(sep='.')[0].strip())
    elif len(fp_filelist[0].split(sep='x')) == 9:
        npars = int(fp_filelist[0].split(sep='x')[-2].strip())

    return npars


def find_fluxfiles(fluxdir, fluxtype, perturbationcode=None, variablelist_files=None, months=None):
    """ Function to find all fluxfiles in a fluxdir, given a list of variables and months.
    Choose between multiple fluxtypes, currently still limited to "CTEHR" and "PARIS" (03-11-2023)."""
    fluxstring = []
    
    if fluxtype == "PARIS":
        if perturbationcode == 'BASE_SS':
            fluxstring += sorted(glob.glob(fluxdir + '/BASE/paris_ctehr_*.nc'))
        else:
            fluxstring += sorted(glob.glob(fluxdir + '/' + perturbationcode + '/paris_ctehr_*.nc'))
    
    elif fluxtype == "CTEHR":
        for var in variablelist_files:
            for mon in months:
                fluxstring += sorted(glob.glob(fluxdir + var + '.' + mon + '.nc'))
    else:
        print('Input fluxtype not recognized, please select one that is defined in the find_fluxfiles() function or define one yourself.')

    return fluxstring


def lazy_load_all_fluxes(fluxfile_list, sim_len=240, start_date=None, end_date=None, fluxtype=None, variablelist_vars=None):
    """ Function to lazy load all fluxes into an xarray Dataset. These fluxes will be used to update the fluxes for each timestep later.
    Choose between multiple fluxtypes, currently still limited to "CTEHR" and "PARIS" (03-11-2023)."""  
    
    variablelist_vars = variablelist_vars + ['time']

    if fluxtype == "PARIS":        
        with xr.open_mfdataset(fluxfile_list) as ds:
           
            if start_date != None and end_date != None:
               
                # Make sure the fluxfiles contain information for the backward simulated hours too!
                start_date_corrected = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(hours=sim_len)
                ds = ds.sel(time=slice(start_date_corrected, end_date))

            return ds[variablelist_vars]

    elif fluxtype == "CTEHR":
        with xr.open_mfdataset(fluxfile_list, combine='by_coords') as ds:
            
            return ds[variablelist_vars]

    else:
        print('Input fluxtype not recognized, please select one that is defined in the find_fluxfiles() function or define one yourself.')


def load_first_flux_timestep(flux_ds, fp_starttime, fp_endtime):
    """ Function to load the fluxes corresponding to the first footprint file in the list into memory,
    which can be updated whilst looping over the other footprints using the update_fluxfiles().
    """

    # Cut the flux_ds to the time range of the first footprint file
    flux_ds = flux_ds.sel(time=slice(fp_endtime, fp_starttime))

    # Load the fluxes into memory in the form of a dictionary
    fluxdict = {}
    for var in flux_ds.variables:
        fluxdict[var] = flux_ds[var][:].compute().values
    
    # Sort dictionairy on dates for later modifications to dict based on order
    fluxdict = dict(sorted(fluxdict.items()))    

    return fluxdict


def update_fluxdict(flux_ds, fluxdict, cur_fp_starttime=None, last_fp_starttime=None):
    """ Function to update a flux array by shifting it in time a given amount of hours (timeshift variable).
    """
    update_dict={}

    timedif = cur_fp_starttime - last_fp_starttime

    # First check if there's a time gap or the files are consecutive
    if timedif <= datetime.timedelta(hours=1):
        new_fluxds = flux_ds.sel(time=cur_fp_starttime)
        for var in flux_ds.variables:
            update_dict[var] = new_fluxds[var][:].compute().values
        fluxdict.update(update_dict)
        del fluxdict[min(fluxdict)]

        logging.info(fluxdict)
        return(fluxdict)
    
    else:
        skipped_times = pd.date_range(last_fp_starttime, cur_fp_starttime, freq='1H')
        delete_times = pd.date_range(min(fluxdict), min(fluxdict) + timedelta(hours=(timedif.total_seconds() / 3600)), freq='1H')
        new_fluxds = flux_ds.sel(time=(min(skipped_times), max(skipped_times) + timedelta(hours=1)))
        for var in flux_ds.variables:
            update_dict[var] = new_fluxds[var][:].compute().values
        fluxdict.update(update_dict)

        for time in delete_times:
            del fluxdict[time]

        logging.info(fluxdict)
        return(fluxdict)
    

def sum_variables(dset, listofvars):
    """ Function to sum all variables in a dictionairy from a list of given variable names. This works for
    dictionairies that are created by the open_fluxfiles() function, and hence originate from an xarray.Dataset() object."""
    data = 0
    for var in listofvars:
        #data = data + dset['data_vars'][var]['data'] # slice of each variable
        data = data + dset[var][:] # slice of each variable

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
    #lat_indices = [lats.index(i) for i in list(
    #    np.round(footprint_df.variables['Latitude'][:].tolist(), 4))]
    #lon_indices = [lons.index(i) for i in list(
    #    np.round(footprint_df.variables['Longitude'][:].tolist(), 4))]

    lat_indices = lats.searchsorted(footprint_df.variables['Latitude'][:])
    lon_indices = lons.searchsorted(footprint_df.variables['Longitude'][:])

    #lat_indices = xr.DataArray(lat_indices, dims=['pos'])
    #lon_indices = xr.DataArray(lon_indices, dims=['pos'])

    return lat_indices, lon_indices


def find_footprint_flux_timeindex(flux_ds, footprint_df, fp_starttime):
    """ Function to find the index of the current timestamp in a given flux file. """
    # First calculate a list of hours since the start of the input footprint file
    hours_relative_to_start_fp = np.array(-1*((footprint_df.variables['Time'][:]).astype(int)))
    
    # Calculate time difference between flux dataset and input footprint file
    #fp_flux_timediff = np.array(flux_ds['coords']['time']['data']) - fp_starttime
    fp_flux_timediff = flux_ds['time'] - np.datetime64(fp_starttime)

    # Extract index of the smallest time difference (which is the timestamp 
    # in the flux dataset closest to the start time of the footprint file)
    #fp_flux_startindex = np.abs(fp_flux_timediff).argmin()
    fp_flux_startindex = np.abs(fp_flux_timediff).argmin().item()

    # Calculate a list of indices of the footprint file relative to the start 
    # of the flux dataset
    fp_flux_diff_indexlist = fp_flux_startindex + hours_relative_to_start_fp
    
    # Create xr.DataArray of indices to use for indexing the fluxfile later
    # hours_into_fluxfile = xr.DataArray(fp_flux_diff_indexlist, dims=['pos'])
    hours_into_fluxfile = fp_flux_diff_indexlist
    
    return hours_into_fluxfile
    

def get_flux_contribution(flux_ds, time_index_list, lat_index_list, lon_index_list, footprint_df, infl_varname, variablelist_vars, 
                         outdir = '/projects/0/ctdas/PARIS/DATA/obspacks/', sum_vars=None, perturbationcode=None):
    """ Function to get the flux contribution of a footprint file, with list indexing. 
    Choose between either a mixed or sector-specific flux contribution.
    """
    if sum_vars==True:
        logging.info('Summing variables!')
        basepath = os.path.abspath(os.path.join(outdir, os.pardir)) + '/BASE/'

        if ((perturbationcode != 'BASE') and (os.path.exists(basepath))):
            if perturbationcode=='HGER':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='ATEN':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='PTEN':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='HFRA':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='DFIN':
                sumvar = ['flux_bio_exchange_prior']
            base_sumvar = variablelist_vars.remove(sumvar)

            with xr.open_dataset(glob.glob(basepath + '*.nc'))[base_sumvar] as base_ds:
                base_cont_perhr_ss = sum_variables(base_ds, listofvars=base_sumvar)

            cont_perhr_ss = dask.delayed(flux_ds[sumvar][time_index_list, lat_index_list,
                                        lon_index_list] * footprint_df[infl_varname][:])
            mixed = (base_cont_perhr_ss + cont_perhr_ss).sum().compute().item()
            return mixed

        elif ((perturbationcode != 'BASE') and not (os.path.exists(basepath))):
            import inspect
            logging.error('"BASE" directory not under the parent directory ' + os.path.abspath(os.path.join(outdir, os.pardir)) + '! Please put it there or fix the path in the ' + inspect.currentframe().f_code.co_name + ' function.')
            
        else:
            sum_dataset = sum_variables(flux_ds, listofvars=variablelist_vars)
            cont_perhr = dask.delayed(sum_dataset[time_index_list, lat_index_list,
                                                    lon_index_list] * footprint_df[infl_varname][:])
            
            mixed = cont_perhr.sum().compute().item()
            return mixed
    
    else:
        logging.info('Not summing variables!')
        cont_ss = [dask.delayed(flux_ds[v][time_index_list, lat_index_list, 
                                                    lon_index_list] * footprint_df[infl_varname][:]).sum() 
                                                    for v in variablelist_vars]
        mixed_list = [result.item() for result in dask.compute(*cont_ss)]
        return mixed_list


def get_flux_contribution_np(flux_ds, time_index_list, lat_index_list, lon_index_list, footprint_df, infl_varname, variablelist_vars, 
                         outdir = '/projects/0/ctdas/PARIS/DATA/obspacks/', sum_vars=None, perturbationcode=None):
    """ Function to get the flux contribution of a footprint file, with list indexing. 
    Choose between either a mixed or sector-specific flux contribution.
    """

    if sum_vars==True:
        logging.info('Summing variables!')
        basepath = os.path.abspath(os.path.join(outdir, os.pardir)) + '/BASE/'

        if ((perturbationcode != 'BASE') and (os.path.exists(basepath))):
            if perturbationcode=='HGER':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='ATEN':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='PTEN':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='HFRA':
                sumvar = ['flux_ff_exchange_prior']
            elif perturbationcode=='DFIN':
                sumvar = ['flux_bio_exchange_prior']
            base_sumvar = variablelist_vars.remove(sumvar)

            with xr.open_dataset(glob.glob(basepath + '*.nc'))[base_sumvar] as base_ds:
                base_cont_perhr_ss = sum_variables(base_ds, listofvars=base_sumvar)

            cont_perhr_ss = flux_ds[time_index_list, lat_index_list,
                                        lon_index_list] * footprint_df.variables[infl_varname][:]
            mixed = (base_cont_perhr_ss + cont_perhr_ss).sum()
            return mixed

        elif ((perturbationcode != 'BASE') and not (os.path.exists(basepath))):
            import inspect
            logging.error('"BASE" directory not under the parent directory ' + os.path.abspath(os.path.join(outdir, os.pardir)) + '! Please put it there or fix the path in the ' + inspect.currentframe().f_code.co_name + ' function.')
        
        else:
            sum_dataset = sum_variables(flux_ds, listofvars=variablelist_vars)
            mixed = (sum_dataset[time_index_list, lat_index_list,
                                                    lon_index_list] * footprint_df[infl_varname][:]).sum()

            return mixed
    
    else:
        logging.info('Not summing variables!')
        mixed_list = [(flux_ds[v][time_index_list, lat_index_list, 
                                                    lon_index_list] * footprint_df[infl_varname][:]).sum() 
                                                    for v in variablelist_vars]
        
        return mixed_list
    

def create_intermediate_dict(dict_obj, cont, background, key, sum_vars=None, perturbationcode=None):   
    if ((sum_vars==True) & (type(cont) != list)):
        # If sum_vars is True, cont itself is the total contribution
        logging.info(f'Mixed flux contribution is {cont}')           
        
        pseudo_obs = background + cont
        values = [background, cont, pseudo_obs]
    else:
        # If sum_vars is False, cont is a list of values, so we have to sum it to get the total contribution
        contribution_mixed = sum(cont)
        logging.info(f'Mixed flux contribution is {contribution_mixed}')

        pseudo_obs = background + contribution_mixed
        values = cont + [background, contribution_mixed, pseudo_obs]

    # Save keys and values in summary_dict
    dict_obj[key] = values

    return dict_obj


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
                fp_starttime = footprint_extract_starttime(file)
                
                logging.info(f'Current ens_mem_num is {i+1}')
                
                # Extract latitude indices from sparse footprint file and insert into xr.DataArray for later indexing
                lat_indices, lon_indices = get_latlon_indices(footprint_df, lats, lons)
                hours_into_file = find_footprint_flux_timeindex(footprint_df=footprint_df, fp_starttime=fp_starttime,
                                                        fp_filelist = fp_filelist)

                logging.info(f'This footprint has {len(lat_indices)} values, has {len(np.unique(hours_into_file))} timesteps and goes {len(np.unique(hours_into_file))} hours backwards in time')

                # Calculate mean background concentration for current footprint
                bg = calculate_mean_bg(fp_starttime=fp_starttime, RDatapath=RDatapath, bgpath=bgpath,
                                    npars=npars, lat=station_lat, lon=station_lon, agl=station_agl, ens_mem_num=i+1)

                logging.info(f'Mean background is {bg}')

                mixed = get_flux_contribution(flux_dataset=flux_dataset, time_index_list=hours_into_file,
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
        

def create_obs_sim_dict(flux_ds, fp_filelist, lats, lons, RDatapath, sim_len,
                        bgpath, station_lat, station_lon, station_agl, stationname,
                        variablelist_vars, stilt_rundir, sum_vars, perturbationcode = None,
                        mp_start_date=None, mp_end_date=None, save_as_pkl=False):
    """ Function to create a dictionary with the background, mixed and pseudo observation. 
    """

    logging.info('Current station is ' + stationname + ', with lat = ' + str(station_lat) + ', lon = ' + str(station_lon) + ', and agl = ' + str(station_agl))
    logging.info('Starting calculation of flux contributions and pseudo observations.')
    
    # Import list of missing footprint files
    missing_fp_filestr = stilt_rundir + stationname + '/missing.footprints'
    if os.path.isfile(missing_fp_filestr):
        missing_fp_file = pd.read_csv(missing_fp_filestr)
        missing_fp_filelist = list(missing_fp_file['ident'])

    summary_dict={}

    # Extract start and end time of first footprint file
    file = fp_filelist[0]
    fp_starttime = footprint_extract_starttime(file)
    fp_endtime = footprint_extract_endtime(file, simulation_len=sim_len)
    
    # Create timerange of simulation
    timerange_sim = pd.date_range(start=mp_start_date, end=mp_end_date, freq='H').to_list()
    timerange_fp = footprint_hours(fp_filelist=fp_filelist, simulation_len=sim_len, shift_forward=False)

    # Extract amount of particles released per STILT simulation from the fp_filelist
    npars = footprint_extract_npars(file)

    # Load the fluxes corresponding to the first footprint file in the list into memory
    fluxdict = load_first_flux_timestep(flux_ds=flux_ds, fp_starttime=fp_starttime, fp_endtime=fp_endtime)

    # Loop over all footprint files
    for simtime in tqdm(timerange_sim):
        logging.info(f'Checking for {simtime} ...')

        if simtime in timerange_fp:            
            logging.info(f'Footprint found for {simtime} !')

            # Calculate index of footprint file in footprint list using footprint_hours
            index_fp = timerange_fp.index(simtime)

            # Get start and end time of footprint
            file = fp_filelist[index_fp]
            ident = os.path.splitext(os.path.basename(file).split(sep='_')[-1])[0]

            # Create keys and values for summary_dict later
            #key = fp_starttime.strftime("%Y-%m-%d %H:%M:%S")
            key = np.datetime64(simtime)
            #key= fp_starttime.to_numpy()
            
            # Calculate mean background concentration for current footprint
            bg = calculate_mean_bg(fp_starttime=simtime, RDatapath=RDatapath, bgpath=bgpath,
                        npars=npars, lat=station_lat, lon=station_lon, agl=station_agl)

            logging.info(f'Mean background is {bg}')

            if os.path.exists(missing_fp_filestr)==False or ident not in missing_fp_filelist:
                with nc.Dataset(file, 'r') as footprint_df:
                    
                    # First update the old fluxdict so that it shifts an X amount in time
                    fluxdict = update_fluxdict(flux_ds=flux_ds, fluxdict=fluxdict, cur_fp_starttime = simtime, last_fp_starttime = last_fp_starttime)

                    # Extract latitude indices from sparse footprint file and insert into xr.DataArray for later indexing
                    lat_indices, lon_indices = get_latlon_indices(footprint_df, lats, lons)
                    hours_into_file = find_footprint_flux_timeindex(flux_ds=fluxdict, footprint_df=footprint_df, 
                                                                    fp_starttime=fp_starttime)

                    logging.info(f'This footprint has {len(lat_indices)} values, has {len(np.unique(hours_into_file))} timesteps and goes {len(np.unique(hours_into_file))} hours backwards in time')

                    # Calculate CTE-HR contribution
                    cont = get_flux_contribution_np(flux_ds=fluxdict, time_index_list=hours_into_file,
                                                        lat_index_list=lat_indices, lon_index_list=lon_indices,
                                                        footprint_df=footprint_df, infl_varname='Influence',
                                                        variablelist_vars=variablelist_vars, sum_vars=sum_vars,
                                                        perturbationcode=perturbationcode)

            elif ident in missing_fp_filelist:
                logging.info(f'Footprint is empty and no surface exchange with particles is recorded, taking background concentration at station location as pseudo observation.')
                if sum_vars==True:
                    logging.info('Summing variables!')
                    cont = 0

                else:
                    logging.info('Not summing variables!')
                    cont = [0] * len(variablelist_vars)
            
            # Calculate pseudo observation from bg and flux contribution            
            summary_dict = create_intermediate_dict(dict_obj=summary_dict, cont=cont, background=bg, key=key, sum_vars=sum_vars, perturbationcode=perturbationcode)
        
            last_fp_starttime = simtime

    logging.info('Finished calculating flux contributions and pseudo observations.')

    if save_as_pkl == True:
        logging.info('Saving dictionary as pickle file')

        # Save dictionary as pickle file
        with open('obs_sim_dict_' + stationname + '_' + fp_starttime.yr + '.pkl', 'wb') as f:
            pickle.dump(summary_dict, f)
    
    elif save_as_pkl == False:
        logging.info('Saving dictionary in Python memory.')

        return summary_dict

def write_dict_to_ObsPack(obspack_list, obs_sim_dict, variablelist_vars, stationcode, start_date = '2021-01-01', end_date = '2022-01-01', 
                          sum_vars = None, mp_start_date = None, mp_end_date = None,
                          outpath='/projects/0/ctdas/PARIS/DATA/obspacks/'):
    """ Function to write a dictionary (that contains the simulation results, the background,
    and pseudo_observations) to a copy of the original ObsPacks in the ObsPack collection.

    Input variables:
    obspack_list: list of ObsPacks that meet the filtering criteria
    obs_sim_dict: dictionary that contains the simulation results, the background, and observations
    stationcode: stationcode of the station
    outpath: path to the output directory
    sum_vars: boolean that indicates whether the variables in the ObsPack file should be summed
    mp_start_date: start date of the current multiprocessing run
    mp_end_date: end date of the current multiprocessing run
    start_date: start date of the simulation
    end_date: end date of the simulation
    """
    
    # Find ObsPack file in list of ObsPacks
    obspack_orig = [s for s in obspack_list if stationcode.lower() in s.lower()][0]
        
    if obspack_orig is None:
        print(f"No ObsPack found for station code {stationcode}")
        return

    # Check if outpath exists, if not, create it
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # Extract multiprocessing timestr to create ObsPack object to save to later
    if mp_start_date != None and mp_end_date != None:
        mp_start_datetime = datetime.strptime(mp_start_date, "%Y-%m-%d")
        mp_end_datetime = datetime.strptime(mp_end_date, "%Y-%m-%d")
        
        timestr = str(mp_start_datetime.strftime("%Y-%m-%d-%H:%M")) + '-' + str(mp_end_datetime.strftime("%Y-%m-%d-%H:%M"))

        # Put simulation results in seperate run-dependent ObsPack files to combine later
        obspack = os.path.join(outpath, 'pseudo_' + os.path.basename(os.path.splitext(obspack_orig)[0]) + '_' + timestr + '.nc')

    else:

        mp_start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        mp_end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

        # Put all results into one obspack file (only one run, so no intermediate files needed)
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
        ds['time'] = pd.to_datetime(ds['time'], unit='s')

        # Define new variables to fill later
        if sum_vars == True:
            new_variables = ['background', 'mixed', 'pseudo_observation']
        else:
            new_variables = variablelist_vars + ['background', 'mixed', 'pseudo_observation']

        for new_variable_name in new_variables:
            dummy_var = xr.DataArray(
                name=new_variable_name,
                dims=ds.time.dims,
                coords=ds.coords
            )
            ds[new_variable_name] = dummy_var
        
        # Resample to hourly data
        logging.info('Resampling to hourly data')

        # Select only the time range of the simulation
        subds = ds.sel(time=slice(mp_start_datetime, mp_end_datetime))
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
        #subds[non_numeric_vars[0]] = pd.to_datetime(subds[non_numeric_vars[0]]) - np.timedelta64(30, 'm')
        #logging.info(subds[non_numeric_vars[0]])

        # Manually resample ObsPack datetime element
        subds['datetime'] = np.array((parse_datetime(ds['datetime']) - pd.to_timedelta('30T')).round('H').strftime("%Y-%m-%dT%H:%M:%S"), dtype='S20')

        # Calculate time differences only once
        #ds_time_seconds = ((subds['time'] - obspack_basetime).values.astype('timedelta64[s]')).astype(np.int64)
        ds_time_seconds = ((subds['time'] - obspack_basetime).values.astype('int64')) / 1e9

        # Loop over all keys and values in dictionary
        logging.info(f'Looping over {len(obs_sim_dict)} footprints')

        logging.info(obs_sim_dict.items())
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
            logging.info(values)

            # Write values to new variables in new ObsPack file
            for v in new_variables:
                subds[v][diff_index] = values[new_variables.index(v)]

    # Save and close the new file
    logging.info(f'Writing results to {obspack}')
    subds.to_netcdf(obspack, mode='w')

def main(filepath, sim_length, fluxdir, stilt_rundir, fluxvarnamelist,  stationsfile, stationcode, outpath, bgfilepath, obspack_list, 
         lats, lons, perturbationcode=None, sum_vars=None, verbose=None, fluxtype=None, start_date=None, end_date=None, mp_start_date = None,
         mp_end_date = None, date_pair=None):
    """ Main function to run the footprint - flux multiplication script. """

    if verbose == True:
        logging.basicConfig(level=logging.INFO, format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=logging.WARNING, format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Get 3D station location
    lat,lon,agl = get_3d_station_location(stationsfile, stationcode, colname_lat='lat', colname_lon='lon',
                                            colname_agl='corrected_alt')  

    # Get list of footprint files for station
    sparse_files = sorted(glob.glob(filepath + 'footprint_' + stationcode + '*.nc'))

    if fluxtype == 'PARIS':
        # Only once per station, create list of all PARIS flux files
        fluxstring = find_fluxfiles(fluxdir = fluxdir, perturbationcode=perturbationcode, fluxtype=fluxtype)

        # Filter list of footprint files by date range (using start_date and end_date, if they exist!)
        mp_start_date, mp_end_date = date_pair
        sparse_files = filter_files_by_date(sparse_files, start_date=mp_start_date, end_date=mp_end_date)

        flux_ds = lazy_load_all_fluxes(fluxstring, sim_len=sim_length, start_date = mp_start_date, 
                                end_date = mp_end_date, variablelist_vars = fluxvarnamelist, 
                                fluxtype=fluxtype)
                    
        # Loop over all footprints files
        obs_sim_dict = create_obs_sim_dict(fp_filelist = sparse_files, flux_ds = flux_ds,
                    lats = lats, lons = lons, RDatapath = filepath, sim_len=sim_length,
                    bgpath = bgfilepath, station_lat = lat, station_lon = lon, 
                    station_agl = agl, stationname = stationcode, variablelist_vars = fluxvarnamelist,
                    stilt_rundir = stilt_rundir, sum_vars=sum_vars, perturbationcode=perturbationcode, 
                    mp_start_date = mp_start_date, mp_end_date = mp_end_date,save_as_pkl=False)

        write_dict_to_ObsPack(obspack_list = obspack_list, obs_sim_dict = obs_sim_dict, sum_vars = sum_vars,
                        variablelist_vars=fluxvarnamelist, stationcode = stationcode, 
                        mp_start_date = mp_start_date, mp_end_date = mp_end_date,
                        start_date= start_date, end_date = end_date, outpath = outpath)
 
    elif fluxtype == 'CTEHR':
        # Define which CTE-HR variables to find fluxfiles for
        fluxfilenamelist = ['nep', 'fire', 'ocean', 'anthropogenic']

        # Extract all unique months between start and end time
        mons = footprint_unique_months(sparse_files, sim_length)

        # Only once per station, create list of all CTE-HR flux files
        fluxstring = find_fluxfiles(fluxdir = fluxdir, variablelist_files = fluxfilenamelist, months = mons, fluxtype=fluxtype)
        
        start_date, end_date = date_pair
        sparse_files = filter_files_by_date(sparse_files, start_date, end_date)

        # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
        flux_ds = lazy_load_all_fluxes(fluxstring, sim_len=sim_length, mp_start_date = start_date, 
                                mp_end_date = end_date, variablelist_vars = fluxvarnamelist, 
                                fluxtype=fluxtype)
        
        # Loop over all footprints files
        obs_sim_dict = create_obs_sim_dict(fp_filelist = sparse_files, flux_ds = flux_ds,
                    lats = lats, lons = lons, RDatapath = filepath, sim_len=sim_length,
                    bgpath = bgfilepath, station_lat = lat, station_lon = lon, 
                    station_agl = agl, stationname = stationcode, variablelist_vars = fluxvarnamelist,
                    stilt_rundir = stilt_rundir, sum_vars = sum_vars, mp_start_date = mp_start_date,
                    mp_end_date = mp_end_date, save_as_pkl=False)

        write_dict_to_ObsPack(obspack_list = obspack_list, obs_sim_dict = obs_sim_dict, sum_vars = sum_vars,
                        variablelist_vars=fluxvarnamelist, stationcode = stationcode, 
                        start_date=start_date, end_date=end_date, outpath = outpath)

    else:
        print("Please provide a valid fluxtype (choose between either 'PARIS' or 'CTEHR' for now)!")

    logging.info(f'Finished writing results to ObsPack file for station {stationcode}')

def compare_ens_members(filepath, sim_length, fluxdir, fluxfilenamelist, fluxvarnamelist, stationsfile,
        bgfilepath, lats, lons, stationcode):
    
    # Get list of footprint files for station
    sparse_files = sorted(glob.glob(filepath + 'footprint_' + stationcode + '*.nc'))

    # Extract all unique months between start and end time
    mons = footprint_unique_months(sparse_files, sim_length)

    # Only once per station, create list of all CTE-HR flux files
    fluxstring = find_fluxfiles(fluxdir = fluxdir, variablelist_files = fluxfilenamelist, months = mons, fluxtype='CTE-HR')

    # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
    cte_ds = open_fluxfiles(fluxstring, sim_len=sim_length, variablelist_vars = fluxvarnamelist)

    # Get 3D station location
    lat,lon,agl = get_3d_station_location(stationsfile, stationcode, colname_lat='lat', colname_lon='lon',
                                            colname_agl='corrected_alt')      
 
    # Loop over all footprints files
    pseudo_df, mixed_df = create_obs_sim_dict_ens(fp_filelist = sparse_files, flux_dataset = cte_ds,
                        lats = lats, lons = lons, RDatapath = filepath, sim_len=sim_length,
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
