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

# ICOS Library
from icoscp.collection import collection
from icoscp.station import station
from icoscp.cpb.dobj import Dobj

stationfile = pd.read_csv('/projects/0/ctdas/PARIS/DATA/stationlist_highestalt.csv',header=0)
stationids = stationfile['code'].tolist()

def get_filenames_zip(obspack_URL):
    """ A function that creates a list of ObsPack filenames from a ZIP archive
    """
									
    zip_archive_path = os.path.join('/data', 'dataAppStorage', 'zipArchive', obspack_URL)
    print(f'Path to your file: {zip_archive_path}\nFile exists: {os.path.exists(zip_archive_path)}')
    
    filelist = []

    # Open the zip file
    with zipfile.ZipFile(zip_archive_path, 'r') as zip_file:
        contents = zip_file.namelist()
        
        for file in contents:
            filelist.append(file)
											
def get_filenames_obspack(dobj_list):
    """ A function that creates a list of ObsPack filenames from a collection 
    on the ICOS Carbon Portal.
    """
    filelist = []
    
    for dobj in dobj_list:
        fileName = dobj.meta.get("fileName")
        filelist.append(fileName)

    return filelist

def unique_stationcodes(dobj_list, case = 'upper'):
    """ A function that returns a list of unique stationcodes from a list 
    of ObsPacks.
    """
    stationlist  = []

    for dobj in dobj_list:
        stationcode = dobj.station['id']
        if case == 'lower':
            if stationcode.lower() not in stationlist:
                stationlist.append(stationcode.lower())
        elif case == 'upper':
            if stationcode.upper() not in stationlist:
                stationlist.append(stationcode.upper())
    
    return stationlist
									
def filter_station_domain(lat, lon, extent =[[-15.0,35.0],[33.0,72.0]]):
    """ Retain only the stations that are within a given domain 
    (in this case CTE-HR domain).
    """
    lat = float(lat)
    lon = float(lon)

    if (lon > extent[0][0] and lon < extent[0][1] and lat > extent[1][0] and lat < extent[1][1]):
        return True
    else:
        return False
    
def filter_station_times(Dobj, start_date = "2021-01-01", end_date = "2022-01-01"):
    """ Retain only the stations that contain data within a given daterange. 
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    start_time = datetime.strptime(Dobj.meta['specificInfo']['acquisition']['interval']['start'], "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.strptime(Dobj.meta['specificInfo']['acquisition']['interval']['stop'], "%Y-%m-%dT%H:%M:%SZ")

    if end_time < start_date or start_time > end_date:
        return False
    else:
        return True

def sampling_height(dobj):
    return dobj.meta['specificInfo']['acquisition']['samplingHeight']
    
def filter_obspacks(collection, extent, start_date = "2021-01-01", end_date = "2022-01-01"):
    """ The main function that performs all filtering operations from other functions
    on the ObsPacks. The filtering includes:
        - highest inlet height
        - within a given domain (in this case the CTE-HR domain)
        - within a given date range
    """
    filtered_list = []

    codelist = unique_stationcodes(collection.data, case = 'upper')
    filenames = get_filenames_obspack(collection.data)

    df = station.getIdList(project='all')
    for stationcode in tqdm(codelist):
        lat = station.get(stationcode, station_df=df).lat
        lon = station.get(stationcode, station_df=df).lon

        if filter_station_domain(lat = lat, lon = lon, extent = extent) == True:
            inletheight_list = []
            subset = [s for s in filenames if stationcode.lower() in s]
                  
            for i in range(0, len(subset)):
                index = filenames.index(subset[i])
                dobj = Dobj(collection.datalink[index])

                if filter_station_times(dobj, start_date = start_date, end_date = end_date) == True:
                    inletheight_list.append(sampling_height(dobj))

            if inletheight_list == []:
                continue
            else:
                index_max = inletheight_list.index(max(inletheight_list))
                index = filenames.index(subset[index_max])
                filtered_list.append(collection.data[index])

    return filtered_list

if __name__ == '__main__':
    
    DOI = '10.18160/PEKQ-M4T1'
    coll = collection.get(DOI)
    codelist = unique_stationcodes(coll.data, case = 'upper')

    ctehr_bbox = [[-15.0,35.0],[33.0,72.0]]
    start_date = "2021-01-01"
    end_date = "2022-01-01"

    filtered_list = filter_obspacks(collection = coll, extent = ctehr_bbox, start_date=start_date, end_date=end_date)
    print(filtered_list)
    print(len(filtered_list))