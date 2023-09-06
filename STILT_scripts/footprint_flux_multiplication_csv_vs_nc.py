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
#fluxdir = '/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
fluxdir = '/gpfs/scratch1/shared/dkivits/fluxes/CTE-HR/modified/'
stationsfile = '/projects/0/ctdas/PARIS/DATA/stationlist_highestalt.csv'
stations = pd.read_csv(stationsfile, header=0)

# Select station
station = 'HTM'
station_loc = [13.4189, 56.0976]

# Simulation length
sim_length = 240

# Define lat/lon grid
lats = np.arange(33, 72, 0.1)
lons = np.arange(-15, 35, 0.2)

testgrid = np.ones((len(lats), len(lons)), dtype=int)
testgrid[0:int(0.5 * len(lats)),:] = 0

# Plot test grid
fig = plt.figure(1, figsize=(16,9))
ax = plt.axes(projection = ccrs.PlateCarree())
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)

im = ax.imshow(testgrid, cmap='Reds', extent=[lons[0],lons[-1],lats[0],lats[-1]])
for i in range(0,len(stations)):
    ax.plot(stations['lon'][i], stations['lat'][i], 'bo', markeredgewidth = 3, markersize=5, transform=ccrs.PlateCarree())
ax.plot(station_loc[0], station_loc[1], 'ro', markeredgewidth = 5, markersize=12, transform=ccrs.PlateCarree())

# Define footprint file list
filepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/'
outpath = '/gpfs/scratch1/shared/dkivits/STILT/footprints_multiplied/'

# Create directory if it does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

csv_files = sorted(glob.glob(filepath+'footprint_' + station + '*.csv'))
nc_files = sorted(glob.glob(filepath+'footprint_' + station + '*.nc'))

#csv_files = [filepath + 'footprint_CES_2021x07x05x16x51.97Nx004.93Ex00200.csv']

timestr_start = csv_files[0][-38:-25]
timestr_end = csv_files[-1][-38:-25]

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
print(cte_ds)

#with dask.config.set(**{'array.slicing.split_large_chunks': True}):
# Loop over all footprints files
for i in range(0,len(csv_files)):
    # Get start and end time of footprint
    csv_file = csv_files[i]
    fp_df_csv = pd.read_csv(csv_file, header=0)

    nc_file = nc_files[i]
    fp_df_nc = xr.open_dataset(nc_file).to_dataframe()

    timelist = fp_df_csv['time'].unique()
    
    # Get start time of footprint
    timestr = csv_file[-38:-25]
    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    
    # Calculate hours since start of CTE-HR flux file to use later for indexing in the for-loop
    time_diff = (fp_starttime - datetime.strptime(mons[0], '%Y%m'))
    hours_since_start_of_ds = time_diff.seconds // 3600 + time_diff.days * 24
    
    # Create an empty list to store the flux*influence values in
    flux_infl_list_csv, flux_infl_list_nc = ([] for i in range(2))
    
    # Loop over all simulated times in footprint file
    for y in range(0,len(timelist)):
        # Select current footprint time 
        hours_into_file = hours_since_start_of_ds - y

        # Group footprint dataframe by time
        fp_curdf_csv = fp_df_csv[fp_df_csv['time'] == timelist[y]]
        fp_curdf_nc = fp_df_nc[fp_df_nc['time'] == timelist[y]]

        # Fill flux_infl_list with values from CTE-HR flux file
        flux_infl_list_csv += (cte_ds.values[hours_into_file,
                                            fp_curdf_csv['index_y']-1, 
                                            fp_curdf_csv['index_x']-1] * fp_curdf_csv['infl']).to_list()
        flux_infl_list_nc += (cte_ds.values[hours_into_file,
                                            fp_curdf_nc['index_y']-1,
                                            fp_curdf_nc['index_x']-1] * fp_curdf_nc['infl']).to_list()
            
    # Add list to fp df
    fp_df_csv['mixed_csv'] = flux_infl_list_csv
    fp_df_csv['mixed_nc'] = flux_infl_list_nc
    
    # Create output file name
    outfile = outpath + os.path.basename(csv_file)[:-4] + '_molefrac_csv_vs_nc.csv'

    # Save fp df to csv
    fp_df_csv.to_csv(outfile, index=False)

# Print total time
print("--- %s seconds ---" % (time.time() - start_time))

"""
start_df = pd.read_csv(csv_files[0], header=0)
timelist = start_df['times'].unique()

fp_curtime_str = fp_curtime.strftime("%Yx%mx%dx%H")
fp_file = glob.glob(filepath + 'footprint_' + station + '_' + fp_curtime_str + '*.csv')[0]

# Loop over footprint files
for i in range(0,len(csv_files)):
    # Get start and end time of footprint
    file = csv_files[i]
    timestr = file[-38:-25]

    df = pd.read_csv(file, header=0)
    timelist = df['times'].unique()

    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    fp_endtime = fp_starttime - timedelta(hours=len(timelist))

    
        fluxfile = [s for s in fluxstring if var in s and curmon in s][0]
        print(fluxfile)

        # Open CTE-HR flux file and extract time
        #CTEHR_nc = nc.Dataset(file, 'r')
        #timevar = CTEHR_nc.variables['time'][:]

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
    timevar = CTEHR_nc.variables['time'][:]
    curtime = basedate + timedelta(seconds=int(timevar[0]))
"""