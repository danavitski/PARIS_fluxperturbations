# Import necessary modules
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import dask
import os 
from functions.fluxfile_functions import *
import time

# Define variable list to loop over later
fluxfilelist = ['nep', 'fire', 'ocean', 'anthropogenic']
fluxnamelist = ['nep', 'fire', 'ocean', 'combustion']
fluxdir = '/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
#fluxdir = '/gpfs/scratch1/shared/dkivits/fluxes/CTE-HR/modified/'

ObsPack_file = '/projects/0/ctdas/PARIS/DATA/obs/co2_bik_tower-insitu_45_allvalid-180magl.nc'
ObsPack_outputfile = os.path.dirname(ObsPack_file) + '/pseudo_' + os.path.basename(ObsPack_file)

#if not os.path.exists(ObsPack_outputfile):
shutil.copyfile(ObsPack_file, ObsPack_outputfile)

ObsPack = nc.Dataset(ObsPack_outputfile, 'a', )
ObsPack.createVariable('pseudo_observation', 'f4', ('time',))
ObsPack.createVariable('background', 'f4', ('time',))
obspack_basetime = datetime(1970,1,1,0,0,0)
obspack_starttime = obspack_basetime + timedelta(seconds=int(ObsPack.variables['time'][0]))

stationsfile = '/projects/0/ctdas/PARIS/DATA/stationlist_highestalt.csv'
stationlist = pd.read_csv(stationsfile, header=0)

# Select station
station = 'CBW'
#station_loc = [13.4189, 56.0976]

# Simulation length
sim_length = 240

# Define lat/lon grid
lats = list(np.round(np.arange(33.05, 72.05, 0.1),4))
lons = list(np.round(np.arange(-14.9, 35.1, 0.2),4))

# Define footprint file list
filepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/'
outpath = '/gpfs/scratch1/shared/dkivits/STILT/footprints_multiplied/'

# Create directory if it does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)


sparse_files = sorted(glob.glob(filepath+'footprint_' + station + '*.nc'))
#dense_files = sorted(glob.glob(filepath+'footprint_' + station + '*.nc'))

#timestr_start = sparse_files[0][-38:-25]
#timestr_end = sparse_files[-1][-38:-25]

timestr_start = sparse_files[0][-37:-24]
timestr_end = sparse_files[-1][-37:-24]

fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')- timedelta(hours=sim_length)
fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H') 

# Extract all unique months between start and end time
mons = pd.date_range(fp_range_start, fp_range_end, freq='D').strftime("%Y%m").unique().tolist()

def add_variables(dset, listofvars):
    data = 0
    for var in listofvars:
        data = data + dset[var][:,:,:] # slice of each variable
    return data 

# Time the script
start_time = time.time()

# Loop over all CTE-HR flux variables
fluxstring = []
for var in fluxfilelist:
    for mon in mons:
        # Define which CTE-HR flux files to loop over
        fluxstring += sorted(glob.glob(fluxdir + var + '.' + mon + '.nc'))

# Check if selection went right
print(fluxstring)

# Open all files in fluxstring as xr_mfdataset
cte_ds = xr.open_mfdataset(fluxstring, combine='by_coords')
cte_ds = add_variables(cte_ds, fluxnamelist)

#with dask.config.set(**{'array.slicing.split_large_chunks': True}):
# Loop over all footprints files
for i in range(0,len(sparse_files)):
    # Get start and end time of footprint
    sparse_file = sparse_files[i]
    #fp_df_sparse = pd.read_csv(sparse_file, header=0)
    fp_df_sparse = nc.Dataset(sparse_file, 'r')

    #dense_file = dense_files[i]
    #fp_df_dense = nc.Dataset(dense_file, 'r')

    timelist = np.unique(fp_df_sparse.variables['Time'][:])
 
    # Get start time of footprint
    #timestr = sparse_file[-38:-25]
    timestr = sparse_file[-37:-24]
    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    obspack_starttime = int((fp_starttime - obspack_basetime).total_seconds())
    #print(obspack_starttime)

    # Calculate hours since start of CTE-HR flux file to use later for indexing in the for-loop
    time_diff = (fp_starttime - datetime.strptime(mons[0], '%Y%m'))
    hours_since_start_of_ds = time_diff.seconds // 3600 + time_diff.days * 24
    
    # Create an empty list to store the flux*Influenceuence values in
    flux_Influence_list_sparse, flux_Influence_list_dense = ([] for i in range(2))

    # Extract latitude indices from sparse footprint file using list comprehension
    lat_indices = [lats.index(i) for i in list(np.round(fp_df_sparse.variables['Latitude'][:].tolist(), 4))]
    lon_indices = [lons.index(i) for i in list(np.round(fp_df_sparse.variables['Longitude'][:].tolist(), 4))]
    
    # Select current footprint time
    hours_into_file = (hours_since_start_of_ds - fp_df_sparse.variables['Time'][:]).astype(int)
    #hours_into_file = hours_since_start_of_ds - fp_df_sparse.variables['Time'][:]

    # Convert the list of indices to xr.DataArrays, so that they can be used for efficient non-ortagonal indexing
    lat_indices = xr.DataArray(lat_indices, dims=['pos'])
    lon_indices = xr.DataArray(lon_indices, dims=['pos'])
    hours_into_file = xr.DataArray(hours_into_file, dims=['pos'])

    # ObsPack.variables['pseudo_observation '][i] = (cte_ds[("pos", hours_into_file),
    #                                                 ("pos", lat_indices),
    #                                                 ("pos", lon_indices)] * fp_df_sparse.variables['Influence'][:]).sum()

    ObsPack.variables['pseudo_observation'][i] = (cte_ds[hours_into_file,
                                                    lat_indices,
                                                    lon_indices] * fp_df_sparse.variables['Influence'][:]).sum()

# Print total time
print("--- %s seconds ---" % (time.time() - start_time))

"""
start_df = pd.read_csv(sparse_files[0], header=0)
timelist = start_df['times'].unique()

fp_curtime_str = fp_curtime.strftime("%Yx%mx%dx%H")
fp_file = glob.glob(filepath + 'footprint_' + station + '_' + fp_curtime_str + '*.csv')[0]

# Loop over footprint files
for i in range(0,len(sparse_files)):
    # Get start and end time of footprint
    file = sparse_files[i]
    timestr = file[-38:-25]

    df = pd.read_csv(file, header=0)
    timelist = df['times'].unique()

    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    fp_endtime = fp_starttime - timedelta(hours=len(timelist))

    
        fluxfile = [s for s in fluxstring if var in s and curmon in s][0]
        print(fluxfile)

        # Open CTE-HR flux file and extract time
        #CTEHR_nc = nc.Dataset(file, 'r')
        #timevar = CTEHR_nc.variables['Time'][:]

        #curtime = basedate + timedelta(seconds=int(timevar[0]))

    # Loop over all times in footprint file
    for y in range(0,len(timelist)):
        curtime = timelist[y]
        subdf = df[df['times'] == curtime]
        
        # Create datetime object of current time
        fp_curtime = fp_starttime - timedelta(hours=y)
        curmon = fp_curtime.strftime("%Y%m")
"""


"""
print(np.shape(array))
print(array[:,:,20])

# test plot footprint
fig = plt.figure(1, figsize=(16,9))
ax = plt.axes(projection = ccrs.PlateCarree())
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)

im = ax.imshow(array[:,:,47], cmap='Reds', extent=[-15,35,33,72])

# Open CTE-HR flux file, extract time to loop over at later stage
    CTEHR_nc = nc.Dataset(file, 'r')
    timevar = CTEHR_nc.variables['Time'][:]
    curtime = basedate + timedelta(seconds=int(timevar[0]))
"""