import pyreadr
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime as dtm
from tqdm import tqdm
#from functions.fluxfile_functions_loadfluxesonebyone import *
from functions.fluxfile_functions import *
import math

def get_last_times(t, path, lat, lon, agl, ens_mem_num=None, npars=250):
    """Get last recorded time of every particle for a certain station, using 
    the RData files from the STILT output.

    Input variables:
    t: datetime object
    path: path to RData files

    Output variables:
    last_times: dataframe with last times of particles at station

    """  
    if (lat <= 0):
        latsign = 'S'
    else:
        latsign = 'N'
    
    if (lon <= 0):
        lonsign = 'W'
    else:
        lonsign = 'E'
    
    #lat = '{:.2f}'.format(abs(lat) // 0.01 * 0.01).zfill(5)
    #lon= '{:.2f}'.format(abs(lon) // 0.01 * 0.01).zfill(6)
    lat = '{:.2f}'.format(round(abs(lat),2)).zfill(5)
    lon= '{:.2f}'.format(round(abs(lon),2)).zfill(6)
    agl = '{:.0f}'.format(int(agl)).zfill(5)
    npars_str = '{:.0f}'.format(npars).zfill(4)
    
    if (ens_mem_num != None):
        ens_mem_num = str(ens_mem_num).zfill(2)
        f = path + '.RData{0}x{1:02}x{2:02}x{3:02}x{4:05}{5:01}x{6:06}{7:01}x{8:05}x{9:04}x{10:02}'
        f = f.format(t.year, t.month, t.day, t.hour, lat, latsign, lon, lonsign, agl, npars_str, ens_mem_num)
    else:
        f = path + '.RData{0}x{1:02}x{2:02}x{3:02}x{4:05}{5:01}x{6:06}{7:01}x{8:05}x{9:04}'
        f = f.format(t.year, t.month, t.day, t.hour, lat, latsign, lon, lonsign, agl, npars_str)
    data = pyreadr.read_r(f)
    varname = f.split('/')[-1][6:]
    foot = data[varname]
    rows = []
    for i in range(1, npars+1):
        rows.append(foot[foot['index']==i].iloc[-1])
    last_times = pd.DataFrame(rows)
    return last_times

def get_hour_idx(t):
    """Get 3-hourly index from datetime object.
    
    Input variables:
    t: datetime object
    
    Output variables:
    3-hourly index
    
    """
    if t.hour >= 22:
        if t.hour == 22 and t.minute > 30 or t.hour == 23:
            return t.hour//3
    if t.hour % 3 == 0 :
        return t.hour//3
    elif t.hour % 3 == 2:
        return t.hour // 3 + 1
    else:
        if t.minute > 30:
            return  t.hour // 3 + 1
        else: return t.hour//3

def get_height_idx(file, lat_idx, lon_idx, time_index, pressure):
    """Get TM5 level for height h.
    
    Input variables:
    file: path to netCDF file
    lat_idx: latitude index
    lon_idx: longitude index
    time_index: time index
    pressure: pressure in Pa
    
    Output variables:
    height_idx: index of height level
    
    """
    with nc.Dataset(file) as ds:
        pressures = ds['pressure'][time_index, :, lat_idx, lon_idx]
    return np.argmin(abs(pressures - pressure))

def get_bg(start, row, bgdir = '/projects/0/ctdas/PARIS/DATA/background/STILT/'):
    """Load background concentration into netCDF Dataset at a certain location and time. 
    GLOBAL_LONS and GLOBAL_LATS are the global lon and lat arrays from the TM5 model.
    
    Input variables:
    start: datetime object
    row: row from last_times dataframe
    
    Output variables:
    bg: background concentration at location and time
    
    """

    # Define GLOBAL_LONS and GLOBAL_LATS
    GLOBAL_LONS = np.arange(-179.5, 180.5, 1)
    GLOBAL_LATS = np.arange(-89.5, 90.5, 1)

    dt = row['time'] / 60 # hours
    time = start + dtm.timedelta(hours=dt)
    bgpath = bgdir + '3d_molefractions_1x1_{}{:02}{:02}.nc'
    bgfile = bgpath.format(time.year, time.month, time.day)
    lat_idx = np.argmin(abs(row['lat'] - GLOBAL_LATS))
    lon_idx = np.argmin(abs(row['lon'] - GLOBAL_LONS))
    time_idx = get_hour_idx(time)
    height_idx = get_height_idx(bgfile, lat_idx, lon_idx, time_idx, row['pres'] * 100)
    with nc.Dataset(bgfile) as ds:
        bg = ds['co2'][time_idx, height_idx, lat_idx, lon_idx]
    return bg

def calculate_mean_bg(fp_starttime, RDatapath, bgpath, npars, lat, lon, agl, ens_mem_num = None):
    """Calculate mean background concentration for a certain footprint.
    Uses the get_last_times() and get_bg() functions. """
    last_times = get_last_times(t = fp_starttime, path = RDatapath, npars = npars, lat = lat, lon = lon, agl = agl, ens_mem_num = ens_mem_num)
    
    bg = []
    for y, row in last_times.iterrows():
        bg.append(get_bg(fp_starttime, row, bgdir = bgpath))
    
    return np.mean(bg)
