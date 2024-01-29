# This file contains functions to link existing footprints
# to ObsPack files.

# Import necessary modules
import pandas as pd
import numpy as np
from numpy import array, logical_and, sqrt
import glob
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import os
import time
from tqdm import tqdm
import logging
import zipfile
import re

def get_files_zip(zip_filepath=str):
    """ A function that creates a list of ObsPack files from a local ZIP archive. 

    Input variables:
    zip_filepath: path to the ZIP archive

    Output variables:
    filelist: list of files in the given ZIP archive

    """

    filelist = []
    filepath = os.path.dirname(zip_filepath) + '/'

    # Open the zip file
    with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
        for member in zip_file.namelist():
            if not os.path.exists(filepath + member):
                zip_file.extractall(filepath)
        
        contents = zip_file.namelist()

        for file in contents:
            if file.endswith('.nc'):
                filelist.append(filepath+file)

    return filelist


def unique_stationcodes(filelist=None, non_obspack_filepath=None, case='upper'):
    """ A function that returns a list of unique stationcodes from a list 
    of ObsPacks.
    """
    stationlist = []

    if non_obspack_filepath != None:
        filelist = glob.glob(non_obspack_filepath + '*.nc')

    for file in filelist:
        stationcode = file.split('/')[-1].split('_')[1]
        if case == 'lower':
            if stationcode.lower() not in stationlist:
                stationlist.append(stationcode.lower())
        elif case == 'upper':
            if stationcode.upper() not in stationlist:
                stationlist.append(stationcode.upper())

    return stationlist


def filter_station_domain(lat, lon, extent=[[-15.0, 35.0], [33.0, 72.0]]):
    """ Retain only the stations that are within a given domain 
    (in this case CTE-HR domain).
    """
    lat = float(lat)
    lon = float(lon)

    if (lon > extent[0][0] and lon < extent[0][1] and lat > extent[1][0] and lat < extent[1][1]):
        return True
    else:
        return False


def filter_station_times(ds, start_date="2021-01-01", end_date="2022-01-01"):
    """ Retain only the stations that contain data within a given daterange. 
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    start_time = datetime.strptime(
        ds.dataset_start_date[:19], "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.strptime(
        ds.dataset_stop_date[:19], "%Y-%m-%dT%H:%M:%S")

    if end_time < start_date or start_time > end_date:
        return False
    else:
        return True


def filter_station_times_nonObsPack(ds, start_date="2021-01-01", end_date="2022-01-01"):
    """ Retain only the stations that contain data within a given daterange. 
    """
    start_date = np.datetime64(datetime.strptime(start_date, "%Y-%m-%d"))
    end_date = np.datetime64(datetime.strptime(end_date, "%Y-%m-%d"))

    start_time = ds.time.values[0]
    end_time = ds.time.values[-1]

    if end_time < start_date or start_time > end_date:
        return False
    else:
        return True


def filter_obspacks(zip_filepath = '/projects/0/ctdas/PARIS/DATA/obs/obspack_co2_466_GVeu_20230913.zip', start_date="2021-01-01", end_date="2022-01-01", extent=[[-15.0, 35.0], [33.0, 72.0]]):
    """ The main function that performs all filtering operations from other functions
    on the ObsPacks. The filtering includes:
        - highest inlet height
        - within a given domain (in this case the CTE-HR domain)
        - within a given date range

    The function returns a list of ObsPacks that meet the filtering criteria.

    Input variables:
    obspack_url: URL of the ObsPack collection ZIP archive on the ICOS CP
    start_date: start date of the filtering period
    end_date: end date of the filtering period
    extent: domain of interest in double-listed format

    Output variables:
    filtered_list: list of ObsPacks that meet the filtering criteria

    """
    filtered_list = []
    
    files = get_files_zip(zip_filepath = zip_filepath)
    codelist = unique_stationcodes(files, case='upper')

    for stationcode in tqdm(codelist):
        inletheight_list = []
        subset = [s for s in files if stationcode.lower() in s]

        for i in range(0, len(subset)):
            with xr.open_dataset(subset[i]) as ds:
                lat = ds.site_latitude
                lon = ds.site_longitude
                agl = ds.dataset_intake_ht
                
                if filter_station_domain(lat=lat, lon=lon, extent=extent) == True:
                    if filter_station_times(ds=ds, start_date=start_date, end_date=end_date) == True:
                        inletheight_list.append(float(agl))

        if inletheight_list != []:
            index_max = inletheight_list.index(max(inletheight_list))
            index = files.index(subset[index_max])
            filtered_list.append(files[index])
        else:
            continue

    return filtered_list

def add_to_obspacklist(obspack_list, non_obspack_filepath, start_date="2021-01-01", end_date="2022-01-01", extent=[[-15.0, 35.0], [33.0, 72.0]]):
    """ Function to add non-ObsPack files to the list of ObsPacks. This is necessary
    if the project includes stations that are not officially recorded in the European ObsPack files. These stations
    have a slightly different structure and therefore need to be loaded into memory in a different way.
    
    PLEASE NOTE: The current implementation of this function is based on the file structure of DECC UK stations (HFD, TTA, VTO, etc). 
    Please add your own external stations and adjust this function according to their file structure! 
    """
    
    filtered_list = []

    # Get list of all files in the directory
    files = glob.glob(non_obspack_filepath + '*.nc')

    # Get list of unique station codes
    codelist = unique_stationcodes(files, case='upper')
    
    for stationcode in tqdm(codelist):
        if stationcode == 'HFD' or stationcode == 'TTA' or stationcode == 'VTO':
            stationcode = stationcode.upper()
        
        inletheight_list = []
        subset = [s for s in files if stationcode in s]

        for i in range(0, len(subset)):
            with xr.open_dataset(subset[i]) as ds:
                lat = ds.station_latitude
                lon = ds.station_longitude
                agl = ds.inlet_height_magl

                if filter_station_domain(lat=lat, lon=lon, extent=extent) == True:
                    if filter_station_times_nonObsPack(ds=ds, start_date=start_date, end_date=end_date) == True:
                        inletheight_list.append(float(agl))
            
        if inletheight_list != []:
            index_max = inletheight_list.index(max(inletheight_list))
            index = files.index(subset[index_max])
            filtered_list.append(files[index])
        
        else:
            continue

    filtered_list = obspack_list + filtered_list

    return filtered_list